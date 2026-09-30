"""Quota-guard tests for image.generate (offline: the HF Space is faked, no quota is spent)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

import config
from image import generate


def _png() -> bytes:
    """A tiny valid PNG."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Isolated evidence dir, a fake reference image and a counting fake Space call."""
    monkeypatch.setattr(config, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(generate, "hf_available", lambda: True)
    ref = tmp_path / "ref.png"
    ref.write_bytes(_png())
    calls = {"n": 0}

    def fake_space(reference_image, prompt, seed, params):
        calls["n"] += 1
        return _png(), {"space": "fake/space", "attempts": 1, "failed_attempts": []}

    monkeypatch.setattr(generate, "_hf_space", fake_space)
    return {"ref": ref, "calls": calls}


def test_identical_request_is_reused(env: dict) -> None:
    """Same (style, prompt, seed, backend, ref) -> existing file, no second call."""
    a = generate.generate_concept("0751471", env["ref"], "prompt A", backend="hf_space")
    b = generate.generate_concept("0751471", env["ref"], "prompt A", backend="hf_space")
    assert env["calls"]["n"] == 1 and not a["reused"] and b["reused"] and a["path"] == b["path"]


def test_per_style_cap_and_force(env: dict) -> None:
    """Third distinct generation for a style is refused; --force bypasses the cap."""
    generate.generate_concept("0751471", env["ref"], "prompt A", backend="hf_space")
    generate.generate_concept("0751471", env["ref"], "prompt B (revision)", backend="hf_space")
    with pytest.raises(generate.GenerationLimit, match="cap is 2"):
        generate.generate_concept("0751471", env["ref"], "prompt C", backend="hf_space")
    assert env["calls"]["n"] == 2
    generate.generate_concept("0751471", env["ref"], "prompt C", backend="hf_space", force=True)
    assert env["calls"]["n"] == 3
    log = generate.read_log("0751471")
    assert [r["forced"] for r in log] == [False, False, True]
    assert all(Path(r["path"]).exists() for r in log)


def test_quota_error_one_retry_then_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quota failure: primary + exactly one fallback attempt, then QuotaExhausted."""
    import gradio_client

    attempts = []

    class FakeClient:
        def __init__(self, space, **kw):
            attempts.append(space)

        def predict(self, **kw):
            raise RuntimeError("You have exceeded your GPU quota (60s requested vs. 0s left). Retry in 23:59:00")

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    ref = tmp_path / "ref.png"
    ref.write_bytes(_png())
    with pytest.raises(generate.QuotaExhausted, match="quota exhausted"):
        generate._hf_space(ref, "p", 42, {"guidance_scale": 2.5, "steps": 28})
    assert attempts == generate.HF_SPACES and len(attempts) == 2


def test_mock_never_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no tokens and no IMAGE_BACKEND, resolution fails loudly instead of using the mock."""
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.setattr(generate, "hf_available", lambda: False)
    monkeypatch.setattr(generate, "replicate_available", lambda: False)
    with pytest.raises(RuntimeError, match="No image backend"):
        generate.resolve_backend(None)
    assert json.dumps(generate.BACKENDS)


def test_run_budget_and_revision_cap(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Within one agent run: max IMAGE_RUN_MAX_REVISIONS revisions and IMAGE_RUN_BUDGET images in total."""
    monkeypatch.setenv("IMAGE_RUN_ID", "run-test")
    monkeypatch.setenv("IMAGE_RUN_BUDGET", "4")
    monkeypatch.setenv("IMAGE_RUN_MAX_REVISIONS", "1")
    generate.generate_concept("0000001", env["ref"], "A", backend="hf_space")
    generate.generate_concept("0000001", env["ref"], "A revised", backend="hf_space")  # the 1 allowed revision
    generate.generate_concept("0000002", env["ref"], "B", backend="hf_space")
    with pytest.raises(generate.GenerationLimit, match="Revision budget"):
        generate.generate_concept("0000002", env["ref"], "B revised", backend="hf_space")
    generate.generate_concept("0000003", env["ref"], "C", backend="hf_space")  # 4th image = budget
    with pytest.raises(generate.GenerationLimit, match="Run budget"):
        generate.generate_concept("0000004", env["ref"], "D", backend="hf_space")
    assert env["calls"]["n"] == 4
    assert [c["is_revision"] for c in generate.list_concepts("0000001")] == [False, True]
