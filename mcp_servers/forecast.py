"""MCP server 'forecast': top-k prediction, SHAP explanations, evaluation report.

Thin FastMCP stdio wrapper over forecasting.select / forecasting.model — no business logic here.
Models and predictions are cached per cutoff (outputs/models, outputs/cache); a cutoff that has
never been run trains once (~1 min), later calls are instant.
Run: python mcp_servers/forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from forecasting import features, model, select  # noqa: E402

mcp = FastMCP("forecast", instructions=(
    "Forecasts next-4-week unit sales per style (product_code) with LightGBM and explains them with SHAP. "
    "'Strong' = high predicted units. Final cutoff = 2020-09-22 (last day of data)."))

Cutoff = Annotated[str, Field(
    description="Last day of data to use, ISO date. '2020-09-22' = all data (forecasts 23 Sep–20 Oct 2020). "
                "Non-Wednesday dates snap back to the last complete Wed→Tue week; no later data is used.",
    examples=["2020-09-22", "2020-05-26"])]


class TopStyle(BaseModel):
    """One selected style with its forecast."""
    rank: int
    product_code: str
    prod_name: str
    product_type_name: str
    garment_group_name: str
    index_group_name: str
    predicted_units_next_4w: float
    units_last_4w: int
    units_last_1w: int
    growth_vs_last_4w: float | None = Field(description="predicted / last-4-week units - 1")
    actual_units_next_4w: float | None = Field(description="known only for past cutoffs (backtest)")


class StyleExplanation(BaseModel):
    """Why the model forecasts what it does for one style."""
    product_code: str
    cutoff: str
    predicted_units_next_4w: float
    naive_run_rate_units: int = Field(description="last week's units × 4 — the drivers are multipliers on this")
    drivers: list[str] = Field(description="top-5 SHAP drivers in plain English, strongest first")


def _cutoff(s: str):
    """Parse and validate a cutoff, turning errors into helpful tool errors."""
    try:
        return select.normalize_cutoff(s)
    except ValueError as e:
        raise ToolError(f"Bad cutoff {s!r}: {e}") from e


def _num(v) -> float | None:
    """NaN-safe float."""
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)


@mcp.tool
def predict_top_k(cutoff: Cutoff = "2020-09-22",
                  k: Annotated[int, Field(ge=1, le=20)] = 3,
                  diversify: Annotated[bool, Field(description="at most one style per garment group")] = True
                  ) -> list[TopStyle]:
    """Return the k styles predicted to sell the most units in the 4 weeks after `cutoff`.

    Only styles that sold in the last 2 weeks are eligible (our only availability proxy — the data has no
    stock). With diversify=True, at most one style per garment group is returned.
    """
    c = _cutoff(cutoff)
    top = select.select_top_k(select.get_predictions(c), k=k,
                              diversify_by="garment_group_name" if diversify else None)
    return [TopStyle(rank=int(r["rank"]), product_code=r["product_code"], prod_name=str(r["prod_name"]),
                     product_type_name=r["product_type_name"], garment_group_name=r["garment_group_name"],
                     index_group_name=r["index_group_name"], predicted_units_next_4w=round(float(r["pred_units"]), 1),
                     units_last_4w=int(r["units_w4"]), units_last_1w=int(r["units_w1"]),
                     growth_vs_last_4w=_num(r["growth"]), actual_units_next_4w=_num(r["y_units"]))
            for _, r in top.iterrows()]


@mcp.tool
def explain_style(product_code: Annotated[str, Field(pattern=r"^\d{6,7}$", description="7-digit style code")],
                  cutoff: Cutoff = "2020-09-22") -> StyleExplanation:
    """Explain a style's forecast: the top-5 SHAP drivers in plain English.

    The model predicts an adjustment to the naive run-rate (last week × 4); each driver says how much a
    feature (momentum, discount, recency, category…) raises or lowers the forecast versus that run-rate.
    """
    c = _cutoff(cutoff)
    code = product_code.zfill(7)
    preds = select.get_predictions(c)
    row = preds[preds["product_code"] == code]
    if row.empty:
        raise ToolError(f"Style {code} did not sell in the 12 weeks before {c}, so it has no forecast. "
                        "Pick a style from predict_top_k.")
    snap = features.make_snapshot(c)
    drivers = model.explain(code, c, snap=snap, booster=select.get_booster(c))
    r = row.iloc[0]
    return StyleExplanation(product_code=code, cutoff=str(c), predicted_units_next_4w=round(float(r["pred_units"]), 1),
                            naive_run_rate_units=int(4 * r["units_w1"]), drivers=drivers)


class EvidenceFiles(BaseModel):
    """Evidence written for one selected style."""
    product_code: str
    forecast_json: str = Field(description="rank, forecast, SHAP drivers, attributes, representative articles")
    sales_curve: str
    ref_images: list[str]


@mcp.tool
def write_evidence(cutoff: Cutoff, product_codes: Annotated[list[str], Field(min_length=1, max_length=3)]
                   ) -> list[EvidenceFiles]:
    """Write the evidence pack for the selected winners (in rank order): outputs/evidence/<code>/forecast.json
    and sales_curve.png, and download their reference photos. Call once after choosing the top-3."""
    try:
        return [EvidenceFiles(**r) for r in select.evidence_for(_cutoff(cutoff), product_codes)]
    except ValueError as e:
        raise ToolError(str(e)) from e


@mcp.tool
def evaluation_report() -> str:
    """Return the model evaluation (markdown): LightGBM vs 3 naive baselines on the validation week and a
    10-cutoff rolling backtest (NDCG@50, precision@k, top-12 hit rate in top-50, WAPE)."""
    p = config.FIGURES_DIR / "eval_table.md"
    if not p.exists():
        raise ToolError("No evaluation yet — run `python -m forecasting.model` first.")
    return p.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(show_banner=False)
