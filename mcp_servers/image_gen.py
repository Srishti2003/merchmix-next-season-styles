"""MCP server 'image': concept generation, CLIP similarity, novelty check, board composition.

Thin FastMCP stdio wrapper over the image/ package — no business logic here.
Generation uses FLUX.1 Kontext [dev] via the free Hugging Face Space (HF_TOKEN, default) or Replicate;
set IMAGE_BACKEND=mock to test the whole flow without either. Quota guards live in image/generate.py.
Run: python mcp_servers/image_gen.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from image import board, generate, similarity  # noqa: E402

mcp = FastMCP("image", instructions=(
    "Generate new product concept images from a winning style's reference photo (FLUX Kontext image edit), "
    "score them with CLIP similarity, check novelty, and compose the final 3-column board."))

ProductCode = Annotated[str, Field(pattern=r"^\d{6,7}$", description="7-digit style code, e.g. '0751471'")]
ImagePath = Annotated[str, Field(description="local file path to a .jpg/.png image")]


class Concept(BaseModel):
    """A generated concept image and how it was made."""
    path: str
    prompt: str
    model: str
    backend: Literal["hf_space", "replicate", "mock"]
    seed: int
    seconds: float
    reused: bool = Field(description="true = identical request already generated; existing file returned, no quota used")


class Novelty(BaseModel):
    """CLIP-based novelty verdict for a concept vs its winning style."""
    product_code: str
    concept_path: str
    max_sim_to_refs: float = Field(description="highest CLIP cosine similarity to the style's own photos")
    sim_to_each_ref: dict[str, float]
    max_sim_to_catalogue_sample: float | None = Field(description="highest similarity to OTHER styles' photos")
    catalogue_sample_size: int
    verdict: Literal["ok", "too_close", "lost_dna"] = Field(
        description="too_close: >=0.80 to a ref (as similar as the style's own colourways → a recolour, not a new "
                    "product); lost_dna: <0.60 (no longer the winning style); else ok")
    thresholds: dict[str, float]


class BoardEntry(BaseModel):
    """One column of the final board."""
    product_code: str
    ref_path: str
    concept_path: str
    why_it_won: str = Field(description="1–2 sentences of sales evidence (forecast, momentum, drivers)")
    what_changed: str = Field(description="1–2 sentences: the 2–3 design changes vs the winning style")
    label: str | None = Field(default=None, description="column heading, e.g. '#1 Pluto RW slacks · Trousers'")
    status: str | None = Field(default=None, description="small tag under the concept, e.g. 'Critic: approved'")
    status_ok: bool = Field(default=True, description="green tag if true, red if false")


def _guard(fn, *args, **kwargs):
    """Run ``fn`` and convert expected failures into readable tool errors."""
    try:
        return fn(*args, **kwargs)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from e


@mcp.tool
def generate_concept(product_code: ProductCode, reference_image: ImagePath,
                     prompt: Annotated[str, Field(min_length=20, description="image-edit instruction (KEEP … CHANGE …)")],
                     seed: int = 42) -> Concept:
    """Generate ONE new product concept by editing the winning style's reference photo with FLUX.1 Kontext [dev].

    Saves outputs/evidence/<code>/concept_<n>.png and appends prompt + params to generation_log.jsonl.
    Takes ~30–90 s. GPU quota is scarce: call it ONCE per brief, plus at most ONE revision per style — a third
    call for the same style is refused. Never retry after a 'quota exhausted' error; report it instead.
    """
    return Concept(**_guard(generate.generate_concept, product_code, reference_image, prompt, seed))


@mcp.tool
def list_concepts(product_code: ProductCode) -> list[dict]:
    """List concepts that ALREADY exist for a style (path, prompt, seed, backend, run_id, is_revision, exists).

    Check this before generate_concept: an existing concept can be sent to the critic first — reusing it costs
    no GPU quota. Generation budget per run: 4 new images, of which at most 1 revision.
    """
    return generate.list_concepts(product_code.zfill(7))


@mcp.tool
def similarity_score(image_a: ImagePath, image_b: ImagePath) -> float:
    """CLIP (ViT-B-32) cosine similarity between two images: ~0.5 unrelated garments, ~0.8+ same kind of
    product, 1.0 identical. A coarse proxy — it sees category, colour and silhouette, not fine details."""
    return round(_guard(similarity.similarity_score, image_a, image_b), 4)


@mcp.tool
def novelty_check(product_code: ProductCode, concept_path: ImagePath) -> Novelty:
    """Check a concept is NEW but still carries the winning style's DNA.

    verdict 'too_close' (>=0.80 CLIP similarity to a reference photo — the level of the style's own colourways,
    i.e. effectively a recolour) → ask for bolder silhouette/detail changes;
    'lost_dna' (<0.60) → the concept drifted away from what sold; 'ok' otherwise.
    """
    return Novelty(**_guard(similarity.novelty_check, product_code, concept_path))


@mcp.tool
def compose_board(entries: Annotated[list[BoardEntry], Field(min_length=1, max_length=3)]) -> str:
    """Render the final 1920-px board (one column per winner: reference → concept, with 'why it won' and
    'what changed' captions) to outputs/final_board.png and return its path."""
    return str(_guard(board.compose_board, [e.model_dump() for e in entries]))


if __name__ == "__main__":
    mcp.run(show_banner=False)
