"""CLIP image similarity (open_clip ViT-B-32, CPU) and a simple novelty check.

CLIP cosine similarity is a weak proxy for "same product": it captures category and silhouette,
not construction details — and it barely reacts to colour. Calibration on our reference photos
(scripts/test_image.py): the SAME style in another colourway scores 0.80–0.94, different styles
0.50–0.76. The plan's 0.90 "too close" cut-off would therefore pass a plain recolour, so TOO_CLOSE
is set to 0.80 (= the lowest colourway-to-colourway similarity): a concept at least as similar
as an existing colourway is a recolour, not a new product. (Greyscale CLIP was tried as a recolour
detector and does not separate the cases: colourways score 0.82–0.93 in greyscale too.)
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np

import config
from data_science import select

CLIP_MODEL, CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
TOO_CLOSE, LOST_DNA = 0.80, 0.60  # see module docstring for the calibration


@lru_cache(maxsize=1)
def _clip():
    """Load (once) the CLIP model and preprocessing transform on CPU."""
    import open_clip
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.eval()
    return model, preprocess


@lru_cache(maxsize=512)
def embed(path: str) -> np.ndarray:
    """L2-normalised CLIP image embedding for an image file (cached by path)."""
    import torch
    from PIL import Image

    model, preprocess = _clip()
    with torch.no_grad():
        x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        v = model.encode_image(x)[0].numpy()
    return v / np.linalg.norm(v)


def similarity_score(image_a: str | Path, image_b: str | Path) -> float:
    """Cosine similarity of two images' CLIP embeddings (≈0.5 unrelated garments … 1.0 identical)."""
    for p in (image_a, image_b):
        if not Path(p).exists():
            raise FileNotFoundError(f"Image not found: {p}")
    return float(embed(str(image_a)) @ embed(str(image_b)))


def catalogue_sample(exclude_code: str, n: int = 200, seed: int = config.SEED) -> list[Path]:
    """Up to ``n`` random reference images of OTHER styles from outputs/refs (whatever is downloaded)."""
    pool = [p for p in sorted(config.REFS_DIR.glob("*/*.jpg")) if p.parent.name != exclude_code]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def novelty_check(product_code: str, concept_path: str | Path) -> dict:
    """Is the concept new but still recognisably the winning style?

    max_sim_to_refs: highest similarity to the style's own reference photos.
    max_sim_to_catalogue_sample: highest similarity to other styles' photos (a copy-of-something-else check).
    verdict: 'too_close' if max_sim_to_refs >= 0.80 (as close as the style's own colourways → recolour/copy),
             'lost_dna' if < 0.60, else 'ok'.
    """
    code = str(product_code).zfill(7)
    refs = select.reference_images(code)
    if not refs:
        raise FileNotFoundError(f"No reference images for {code} in {config.REFS_DIR / code}.")
    sims = {p.name: similarity_score(p, concept_path) for p in refs}
    cat = catalogue_sample(code)
    cat_sims = [similarity_score(p, concept_path) for p in cat]
    best = max(sims.values())
    verdict = "too_close" if best >= TOO_CLOSE else "lost_dna" if best < LOST_DNA else "ok"
    return {
        "product_code": code, "concept_path": str(concept_path),
        "max_sim_to_refs": round(best, 4), "sim_to_each_ref": {k: round(v, 4) for k, v in sims.items()},
        "max_sim_to_catalogue_sample": round(max(cat_sims), 4) if cat_sims else None,
        "catalogue_sample_size": len(cat), "verdict": verdict,
        "thresholds": {"too_close": TOO_CLOSE, "lost_dna": LOST_DNA},
    }
