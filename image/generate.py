"""Concept image generation with FLUX.1 Kontext [dev]: free Hugging Face Space (default), Replicate, or a mock.

Backends (``backend`` argument, else env IMAGE_BACKEND, else the first usable one):
- "hf_space"  (default) black-forest-labs/FLUX.1-Kontext-Dev via gradio_client, endpoint /infer.
              Free ZeroGPU quota is small (~5–10 images/day), so quota is guarded (see below).
              If the primary Space fails (quota/queue/runtime error), ONE retry goes to the fallback
              Space zerogpu-aoti/FLUX.1-Kontext-Dev. No other retries, no loops.
- "replicate" black-forest-labs/flux-kontext-dev (needs REPLICATE_API_TOKEN, ~$0.03/image). Optional.
- "mock"      local stand-in, only when asked for explicitly — a missing token never silently yields fakes.

Quota guards (all bypassed only by ``force=True``, which the MCP tool never passes):
1. Idempotent: the same (style, prompt, seed, backend) returns the existing concept instead of re-generating.
2. At most MAX_PER_STYLE real (non-mock) generations per style — the first concept + one revision.
3. Per agent run (env IMAGE_RUN_ID): at most IMAGE_RUN_BUDGET new images (default 4) of which at most
   IMAGE_RUN_MAX_REVISIONS (default 1) are revisions (= a style that already had a concept).
Every generation is appended to outputs/evidence/<code>/generation_log.jsonl (prompt, params, backend, output).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import config

REPLICATE_MODEL = "black-forest-labs/flux-kontext-dev"
HF_SPACES = ["black-forest-labs/FLUX.1-Kontext-Dev", "zerogpu-aoti/FLUX.1-Kontext-Dev"]  # primary, fallback
HF_ENDPOINT = "/infer"  # from Client(...).view_api(): (input_image, prompt, seed, randomize_seed, guidance_scale, steps)
BACKENDS = ("hf_space", "replicate", "mock")
MAX_PER_STYLE = 2


class QuotaExhausted(RuntimeError):
    """The free Hugging Face GPU quota is used up (or both Spaces are unavailable)."""


class GenerationLimit(RuntimeError):
    """A local guard refused to spend quota (per-style cap reached)."""


def replicate_available() -> bool:
    """True if a real-looking Replicate token is configured (not the .env.example placeholder)."""
    tok = os.getenv("REPLICATE_API_TOKEN") or config.REPLICATE_API_TOKEN or ""
    return tok.startswith("r8_") and len(tok) > 20


def hf_available() -> bool:
    """True if a real-looking Hugging Face token is configured."""
    tok = os.getenv("HF_TOKEN") or config.HF_TOKEN or ""
    return tok.startswith("hf_") and len(tok) > 20


def resolve_backend(backend: str | None = None) -> str:
    """Pick the generation backend; raise a helpful error if none is usable."""
    backend = backend or os.getenv("IMAGE_BACKEND")
    if backend and backend not in BACKENDS:
        raise RuntimeError(f"Unknown IMAGE_BACKEND {backend!r}; use one of {BACKENDS}.")
    if backend == "replicate" and not replicate_available():
        raise RuntimeError("IMAGE_BACKEND=replicate but REPLICATE_API_TOKEN is missing or a placeholder in .env.")
    if backend == "hf_space" and not hf_available():
        raise RuntimeError("IMAGE_BACKEND=hf_space but HF_TOKEN is missing in .env (https://huggingface.co/settings/tokens).")
    if backend:
        return backend
    if hf_available():
        return "hf_space"
    if replicate_available():
        return "replicate"
    raise RuntimeError("No image backend: set HF_TOKEN (free HF Space) or REPLICATE_API_TOKEN in .env, "
                       "or pass backend='mock' / set IMAGE_BACKEND=mock for a local test image.")


def _next_concept_path(out_dir: Path) -> Path:
    """outputs/evidence/<code>/concept_<n>.png with the next free n (1-based)."""
    n = 1
    while (out_dir / f"concept_{n}.png").exists():
        n += 1
    return out_dir / f"concept_{n}.png"


def run_generations(run_id: str) -> list[dict]:
    """All generation records (any style) made during agent run ``run_id``."""
    out = []
    for log in config.EVIDENCE_DIR.glob("*/generation_log.jsonl"):
        out += [r for r in (json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip())
                if r.get("run_id") == run_id]
    return out


def list_concepts(code: str) -> list[dict]:
    """Existing concepts for a style (oldest first): path, prompt, seed, backend, ts, run_id, exists."""
    return [{"path": r["path"], "prompt": r["prompt"], "seed": r["seed"], "backend": r["backend"], "ts": r["ts"],
             "run_id": r.get("run_id"), "is_revision": r.get("is_revision", False), "exists": Path(r["path"]).exists()}
            for r in read_log(code)]


def read_log(code: str) -> list[dict]:
    """All generation records for a style (empty if none)."""
    log = config.EVIDENCE_DIR / str(code).zfill(7) / "generation_log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _to_png(raw: bytes | Path) -> bytes:
    """Normalise any image (webp/jpg/png, bytes or file) to PNG bytes."""
    img = Image.open(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _is_quota_error(msg: str) -> bool:
    """Heuristic: does a Space error message mean the ZeroGPU quota is exhausted?"""
    m = msg.lower()
    return any(s in m for s in ("quota", "exceeded your gpu", "zerogpu", "retry in", "rate limit"))


def _hf_space(reference_image: Path, prompt: str, seed: int, params: dict) -> tuple[bytes, dict]:
    """Call the FLUX Kontext Space (primary, then ONE retry on the fallback). Returns (png, info)."""
    from gradio_client import Client, handle_file

    errors = []
    for space in HF_SPACES:  # exactly 2 attempts max: primary + one fallback retry
        try:
            client = Client(space, token=os.getenv("HF_TOKEN") or config.HF_TOKEN, verbose=False)
            result, used_seed = client.predict(
                input_image=handle_file(str(reference_image)), prompt=prompt, seed=seed, randomize_seed=False,
                guidance_scale=params["guidance_scale"], steps=params["steps"], api_name=HF_ENDPOINT)
            path = result["path"] if isinstance(result, dict) else result
            return _to_png(Path(path)), {"space": space, "returned_seed": used_seed, "attempts": len(errors) + 1,
                                         "failed_attempts": errors}
        except Exception as e:  # noqa: BLE001 — Space errors come as many types (AppError, ValueError, httpx…)
            errors.append(f"{space}: {type(e).__name__}: {str(e)[:300]}")
    if any(_is_quota_error(e) for e in errors):
        raise QuotaExhausted("Hugging Face GPU quota exhausted — no image was generated. The free ZeroGPU quota "
                             "(~5–10 images/day) resets over ~24 h; try later, use a HF Pro token, or set "
                             "IMAGE_BACKEND=replicate. Details: " + " | ".join(errors))
    raise RuntimeError("Both FLUX Kontext Spaces failed (no retries left): " + " | ".join(errors))


def _replicate(reference_image: Path, prompt: str, seed: int, params: dict) -> bytes:
    """Call FLUX Kontext [dev] on Replicate and return PNG bytes."""
    import replicate

    with open(reference_image, "rb") as f:
        out = replicate.run(REPLICATE_MODEL, input={"input_image": f, "prompt": prompt, "seed": seed,
                                                    "aspect_ratio": "match_input_image", "output_format": "png",
                                                    "guidance": params["guidance_scale"],
                                                    "num_inference_steps": params["steps"]})
    out = out[0] if isinstance(out, (list, tuple)) else out
    return _to_png(out.read() if hasattr(out, "read") else bytes(out))


def _mock(reference_image: Path, prompt: str, seed: int) -> bytes:
    """Deterministic stand-in: the reference, mirrored and colour-shifted, stamped MOCK.

    Keeps the garment's shape (so CLIP similarity is high but < 1) — enough to exercise
    similarity, novelty checks and board layout without an API call.
    """
    img = ImageOps.exif_transpose(Image.open(reference_image)).convert("RGB")
    h = int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest(), 16)
    r, g, b = img.split()
    shift = [r, g, b][h % 3], [r, g, b][(h // 3) % 3], [r, g, b][(h // 9) % 3]
    img = Image.merge("RGB", shift)
    img = ImageOps.mirror(ImageOps.autocontrast(img, cutoff=1))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", max(24, img.width // 12))
    except OSError:
        font = ImageFont.load_default()
    d.text((img.width * 0.05, img.height * 0.03), "MOCK CONCEPT", fill=(220, 40, 40), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_concept(product_code: str, reference_image: str | Path, prompt: str, seed: int = 42,
                     backend: str | None = None, force: bool = False,
                     guidance_scale: float = 2.5, steps: int = 28, note: str | None = None) -> dict:
    """Generate one concept image from a reference photo + edit prompt.

    Saves outputs/evidence/<code>/concept_<n>.png and logs the call. Returns
    {path, prompt, model, backend, seed, seconds, reused}. Without ``force``: an identical earlier
    request returns the existing file (reused=True), and a style with MAX_PER_STYLE real generations
    already raises GenerationLimit instead of spending quota.
    """
    code = str(product_code).zfill(7)
    ref = Path(reference_image)
    if not ref.exists():
        raise FileNotFoundError(f"Reference image not found: {ref}")
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    backend = resolve_backend(backend)
    params = {"guidance_scale": guidance_scale, "steps": steps}
    keys = ("path", "prompt", "model", "backend", "seed", "seconds")

    history = read_log(code)
    run_id = os.getenv("IMAGE_RUN_ID")
    is_revision = any(Path(r["path"]).exists() and (r["backend"] == "mock") == (backend == "mock") for r in history)
    if not force:
        for rec in reversed(history):
            same = (rec["prompt"], rec["seed"], rec["backend"], rec["reference_image"]) == (prompt, seed, backend, str(ref))
            if same and Path(rec["path"]).exists():
                return {**{k: rec[k] for k in keys}, "reused": True}
        real = [r for r in history if r["backend"] != "mock"]
        if backend != "mock" and len(real) >= MAX_PER_STYLE:
            raise GenerationLimit(f"Style {code} already has {len(real)} generated concepts "
                                  f"({', '.join(Path(r['path']).name for r in real)}); the cap is {MAX_PER_STYLE} "
                                  "(first concept + one revision) to protect the free GPU quota. "
                                  "Pick the best existing concept, or re-run by hand with --force.")
        if run_id:
            done = run_generations(run_id)
            budget = int(os.getenv("IMAGE_RUN_BUDGET", "4"))
            max_rev = int(os.getenv("IMAGE_RUN_MAX_REVISIONS", "1"))
            if len(done) >= budget:
                raise GenerationLimit(f"Run budget reached: {len(done)}/{budget} new images already generated in run "
                                      f"{run_id}. Use the existing concepts.")
            if is_revision and sum(bool(r.get("is_revision")) for r in done) >= max_rev:
                raise GenerationLimit(f"Revision budget reached ({max_rev} per run): {code} already has a concept. "
                                      "Keep the best existing concept and report the critic's note instead.")

    out_dir = config.EVIDENCE_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    info: dict = {}
    if backend == "hf_space":
        png, info = _hf_space(ref, prompt, seed, params)
        model = f"FLUX.1 Kontext [dev] — HF Space {info['space']}"
    elif backend == "replicate":
        png, model = _replicate(ref, prompt, seed, params), f"FLUX.1 Kontext [dev] — Replicate {REPLICATE_MODEL}"
    else:
        png, model, params = _mock(ref, prompt, seed), "mock (mirrored + colour-shifted reference)", {}
    path = _next_concept_path(out_dir)
    path.write_bytes(png)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "product_code": code,
        "reference_image": str(ref), "prompt": prompt, "seed": seed, "backend": backend, "model": model,
        "params": params, "path": str(path), "seconds": round(time.time() - t, 2), "forced": force,
        "run_id": run_id, "is_revision": is_revision, "note": note, **info,
    }
    with open(out_dir / "generation_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {**{k: rec[k] for k in keys}, "reused": False}
