"""Pydantic response models for the API (examples are abridged real values from outputs/predictions.json)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(BaseModel):
    product_type: str | None = Field(description="H&M product type, e.g. Trousers")
    garment_group: str | None = Field(description="H&M garment group; the top-3 has at most one style per group")


class WeekUnits(BaseModel):
    week_start: date = Field(description="Wednesday that starts the Wed→Tue week")
    units: int


class WeekSales(WeekUnits):
    buyers: int


class ShortHistory(BaseModel):
    last_8_weeks: list[WeekUnits] = Field(description="weekly units, oldest first, ending the week before the cutoff")
    units_last_4w: int = Field(description="units sold in the 4 weeks before the cutoff")


class ForecastWindow(BaseModel):
    start: date
    end: date


class StyleSummary(BaseModel):
    style_id: str = Field(description="product_code: 7 digits, all colour variants of one design")
    name: str | None
    rank: int | None = Field(description="1-3 for the selected winners, null for every other style")
    forecast_rank: int = Field(description="position by forecast units among all scored styles")
    prediction_score: float = Field(description="calibrated P(style is a top-0.1% seller over the next 4 weeks)")
    confidence_top1pct: float = Field(description="calibrated P(style is a top-1% seller over the next 4 weeks)")
    forecast_units: float = Field(description="regressor forecast of units in the forecast window")
    category: Category
    sales_history: ShortHistory
    image_url: str | None = Field(description="catalogue photo of the best-selling colourway; null if not downloaded")


class TopStylesResponse(BaseModel):
    cutoff: date = Field(description="prediction cutoff: data up to the day before is used")
    forecast_window: ForecastWindow
    total: int = Field(description="number of styles available in the list")
    limit: int
    offset: int
    styles: list[StyleSummary]

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
        "total": 200, "limit": 1, "offset": 0,
        "styles": [{"style_id": "0751471", "name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
                    "prediction_score": 1.0, "confidence_top1pct": 1.0, "forecast_units": 7417.3,
                    "category": {"product_type": "Trousers", "garment_group": "Trousers"},
                    "sales_history": {"last_8_weeks": [{"week_start": "2020-09-16", "units": 1711}],
                                      "units_last_4w": 9182},
                    "image_url": "/images/refs/0751471/0751471001.jpg"}]}]})


class Reason(BaseModel):
    text: str = Field(description="the SHAP driver in plain words")
    feature: str | None
    value: str | None
    direction: Literal["raises", "lowers"] | None
    effect_pct: float | None = Field(description="change vs the last-week × 4 run-rate, in percent")
    raw: str = Field(description="the driver as stored in predictions.json")


class Explanation(BaseModel):
    why_selected: str = Field(description="one line: why the style is (or is not) in the top 3")
    reasons: list[Reason] = Field(description="top-5 TreeSHAP drivers of the forecast")
    source: str


class KeepTrait(BaseModel):
    trait: str
    evidence: str | None = None


class ChangeAxis(BaseModel):
    axis: str
    from_: str | None = Field(None, alias="from", serialization_alias="from")
    to: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class CriticStatus(BaseModel):
    decision: str | None = Field(description="critic decision on the final concept: approve or revise")
    status: str | None = Field(description="status shown on the final board")
    note: str | None
    max_similarity_to_references: float | None = Field(description="CLIP similarity to the reference photos")
    changes_not_visible: list[str] = Field(description="brief changes the critic could not see in the image")


class Concept(BaseModel):
    image_url: str | None
    reference_image_url: str | None
    keep: list[KeepTrait]
    change: list[ChangeAxis]
    what_changed: str | None
    critic: CriticStatus


class StyleDetail(BaseModel):
    style_id: str
    name: str | None
    rank: int | None
    forecast_rank: int
    prediction_score: float
    confidence_top1pct: float
    p_top0_1pct: float
    forecast_units: float
    category: Category
    attributes: dict[str, str | int | None] = Field(description="product information from articles.csv")
    performance: dict[str, float | int | None] = Field(description="recent sales performance before the cutoff")
    explanation: Explanation
    sales_history_26w: list[WeekSales] = Field(description="up to 26 weeks before the cutoff (from first sale)")
    image_url: str | None
    reference_image_urls: list[str] = Field(description="downloaded catalogue photos of this style")
    concept: Concept | None = Field(description="the generated next-season concept (top-3 only)")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "style_id": "0751471", "name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
        "prediction_score": 1.0, "confidence_top1pct": 1.0, "p_top0_1pct": 1.0, "forecast_units": 7417.3,
        "category": {"product_type": "Trousers", "garment_group": "Trousers"},
        "attributes": {"colour_group_name": "Black", "n_colours": 10},
        "performance": {"units_last_4w": 9182, "units_same_4w_last_year": 3159},
        "explanation": {"why_selected": "Selected #1: highest forecast in Trousers (7,417 units for 2020-09-23 to "
                                        "2020-10-20), sold in the last 2 weeks; P(top 0.1%) = 1.00.",
                        "reasons": [{"text": "Discount vs highest price seen is 2%: this raises the forecast by 21% "
                                             "compared with repeating last week's sales for 4 weeks.",
                                     "feature": "discount vs highest price seen", "value": "2%",
                                     "direction": "raises", "effect_pct": 21.0,
                                     "raw": "discount vs highest price seen = 2% → raises the forecast ×1.21 vs the "
                                            "last-week run-rate"}],
                        "source": "outputs/evidence/0751471/forecast.json"},
        "sales_history_26w": [{"week_start": "2020-09-16", "units": 1711, "buyers": 1178}],
        "image_url": "/images/refs/0751471/0751471001.jpg",
        "reference_image_urls": ["/images/refs/0751471/0751471001.jpg"],
        "concept": {"image_url": "/images/evidence/0751471/concept_2.png",
                    "reference_image_url": "/images/refs/0751471/0751471001.jpg",
                    "keep": [{"trait": "slim tapered ankle-length cigarette leg"}],
                    "change": [{"axis": "pattern", "from": "solid black",
                                "to": "brushed wool-look houndstooth check in charcoal and camel"}],
                    "what_changed": "Charcoal-and-camel glen-check wool-look fabric; camel contrast side stripe.",
                    "critic": {"decision": "approve", "status": "Critic: approved", "note": "...",
                               "max_similarity_to_references": 0.7263, "changes_not_visible": []}}}]})


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    prediction_cutoff: date
    forecast_window: ForecastWindow
    n_styles: int = Field(description="styles available through the API")
    n_styles_scored: int = Field(description="styles scored by the model at the cutoff")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "status": "ok", "model_version": "regressor-20200923-r94+clf-top1pct-r294+clf-top0.1pct-r135",
        "prediction_cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
        "n_styles": 200, "n_styles_scored": 20318}]})


class ErrorResponse(BaseModel):
    detail: str
    style_id: str | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"detail": "Style 0000001 is not in the published predictions (top 200 styles by forecast units).",
         "style_id": "0000001"}]})


class ValidationErrorResponse(BaseModel):
    detail: str
    errors: list[dict]

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"detail": "Invalid request parameters.",
         "errors": [{"loc": ["query", "limit"], "msg": "Input should be less than or equal to 200",
                     "type": "less_than_equal"}]}]})
