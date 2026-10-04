"""M6 manual test: CLIP calibration, ONE hand-written concept on ONE winner, novelty check, board render.

  python scripts/test_image.py                         # mock backend, outputs in outputs/mock_run/
  python scripts/test_image.py --backend hf_space      # 1 real generation (free HF Space quota!)
  python scripts/test_image.py --backend hf_space --force   # regenerate even if this concept exists

Real backends make exactly one generation call (for --code); the board test runs only with the mock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = {
    "0751471": ("Redesign this garment as a new next-season product. Keep: slim tapered ankle-length trouser "
                "silhouette, regular waist with belt loops, front slant pockets, pressed centre crease. "
                "Change: fabric to a brushed wool-look check in charcoal and camel, add a narrow satin side "
                "stripe, cropped split hem. Studio product photo, plain light-grey background, same camera "
                "angle, no model, no text."),
}


def parse() -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "hf_space", "replicate"], default="mock")
    ap.add_argument("--code", default="0751471", help="winner to test (must have a prompt in PROMPTS)")
    ap.add_argument("--force", action="store_true", help="bypass the no-regenerate and per-style quota guards")
    ap.add_argument("--ref", help="input image to edit (default: the style's best-selling reference photo)")
    ap.add_argument("--prompt-file", help="text file with the prompt (default: PROMPTS[code])")
    ap.add_argument("--note", help="free-text note stored in generation_log.jsonl (e.g. who approved a manual extra)")
    return ap.parse_args()


def side_by_side(ref: str, concept: str, out: Path, caption: str) -> Path:
    """Save reference and concept next to each other with a caption line."""
    from PIL import Image, ImageDraw, ImageOps

    from image.board import _font

    a, b = (ImageOps.contain(Image.open(p).convert("RGB"), (600, 900)) for p in (ref, concept))
    canvas = Image.new("RGB", (a.width + b.width + 60, max(a.height, b.height) + 90), "#fcfcfb")
    canvas.paste(a, (20, 20))
    canvas.paste(b, (a.width + 40, 20))
    d = ImageDraw.Draw(canvas)
    d.text((20, max(a.height, b.height) + 35), caption, font=_font(22), fill="#0b0b0b")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def main() -> None:
    """Run the M6 checks and print a short report."""
    args = parse()
    if args.backend == "mock":
        os.environ["HM_EVIDENCE_DIR"] = str(ROOT / "outputs" / "mock_run" / "evidence")
    import config
    from data_science import select
    from image import board, generate, similarity

    print(f"backend = {args.backend} | HF token: {generate.hf_available()} | "
          f"Replicate token: {generate.replicate_available()}")

    # 1) CLIP calibration on the downloaded reference photos
    refs = {c: select.reference_images(c) for c in ("0751471", "0762846", "0685814")}
    if args.code not in refs:
        refs[args.code] = select.reference_images(args.code)
    same = [similarity.similarity_score(a, b) for ps in refs.values() for a, b in combinations(ps, 2)]
    cross = [similarity.similarity_score(a, b) for c1, c2 in combinations(refs, 2) for a in refs[c1] for b in refs[c2]]
    print(f"CLIP calibration: same style, other colour {min(same):.3f}–{max(same):.3f} (n={len(same)}); "
          f"different styles {min(cross):.3f}–{max(cross):.3f} (n={len(cross)})")

    # 2) ONE hand-written prompt on ONE winner
    code = args.code
    ref = args.ref or str(refs[code][0])
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else PROMPTS[code]
    out = generate.generate_concept(code, ref, prompt, seed=42, backend=args.backend, force=args.force,
                                    note=args.note)
    print(f"\nconcept -> {out['path']}\n  model: {out['model']} | {out['seconds']}s | reused existing: {out['reused']}")
    nov = similarity.novelty_check(code, out["path"])
    print("novelty_check:", json.dumps({k: nov[k] for k in ("max_sim_to_refs", "sim_to_each_ref",
                                                            "max_sim_to_catalogue_sample", "verdict")}, indent=1))
    sbs = side_by_side(ref, out["path"], config.EVIDENCE_DIR.parent / "figures" / f"m6_{code}_ref_vs_concept.png",
                       f"{code}: reference (left) vs concept (right) — CLIP max sim to refs "
                       f"{nov['max_sim_to_refs']:.3f} → {nov['verdict']}")
    print(f"side-by-side -> {sbs}")

    # 3) board with placeholder text — mock only (never spends real quota on placeholders)
    if args.backend == "mock":
        entries = []
        for code_i, label in [("0751471", "#1 Pluto RW slacks · Trousers"), ("0762846", "#2 Lucy blouse · Blouses"),
                              ("0685814", "#3 RICHIE HOOD · Jersey Basic")]:
            concept = out["path"] if code_i == code else generate.generate_concept(
                code_i, refs[code_i][0], f"placeholder prompt for {code_i} — mock board test", backend="mock")["path"]
            entries.append({"product_code": code_i, "ref_path": str(refs[code_i][0]), "concept_path": concept,
                            "label": label, "why_it_won": "Placeholder why-it-won text.",
                            "what_changed": "Placeholder what-changed text."})
        print(f"board -> {board.compose_board(entries, out_path=config.OUT_DIR / 'mock_run' / 'board_mock.png')}")


if __name__ == "__main__":
    main()
