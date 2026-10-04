"""Pydantic response models for the API (examples are abridged real values from outputs/predictions.json)."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(BaseModel):
    product_type: str | None = Field(description="H&M product type, e.g. Trousers")
    garment_group: str | None = Field(description="H&M garment group; the top-3 has at most one style per group")
    index_group: str | None = Field(None, description="H&M index group, e.g. Ladieswear, Menswear, Divided")


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
    name: str | None = Field(description="display name (trailing copy number such as ' (1)' removed)")
    raw_name: str | None = Field(description="product name as in articles.csv")
    rank: int | None = Field(description="1-3 for the selected winners, null for every other style")
    forecast_rank: int = Field(description="position by forecast units among all scored styles")
    prediction_score: float = Field(description="calibrated P(style is a top-0.1% seller over the next 4 weeks)")
    confidence_top1pct: float = Field(description="calibrated P(style is a top-1% seller over the next 4 weeks)")
    forecast_units: float = Field(description="regressor forecast of units in the forecast window")
    category: Category
    sales_history: ShortHistory
    image_url: str | None = Field(description="catalogue photo of the best-selling colourway; null if not downloaded")
    actual_units: int | None = Field(None, description="observed units in the window (observed seasons only)")
    actual_rank: int | None = Field(None, description="rank by observed units (observed seasons only)")


class TopStylesResponse(BaseModel):
    season: str
    season_label: str
    observed: bool = Field(description="true when the forecast window is in the data (actuals available)")
    cutoff: date = Field(description="prediction cutoff: data up to the day before is used")
    forecast_window: ForecastWindow
    total: int = Field(description="number of styles available in the list")
    n_styles_scored: int = Field(description="number of styles the model scored at this cutoff")
    limit: int
    offset: int
    styles: list[StyleSummary]

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "season": "AW2020", "season_label": "Autumn/Winter 2020 (forecast)", "observed": False,
        "cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
        "total": 200, "n_styles_scored": 20318, "limit": 1, "offset": 0,
        "styles": [{"style_id": "0751471", "name": "Pluto RW slacks", "raw_name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
                    "prediction_score": 1.0, "confidence_top1pct": 1.0, "forecast_units": 7417.3,
                    "category": {"product_type": "Trousers", "garment_group": "Trousers", "index_group": "Ladieswear"},
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


class Actual(BaseModel):
    units: int | None
    rank: int | None = Field(description="rank by observed units among all scored styles")
    weekly: list[WeekUnits]


class StyleDetail(BaseModel):
    season: str
    cutoff: date
    forecast_window: ForecastWindow
    style_id: str
    name: str | None = Field(description="display name (trailing copy number such as ' (1)' removed)")
    raw_name: str | None = Field(description="product name as in articles.csv")
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
    concept: Concept | None = Field(description="the generated next-season concept (AW2020 top-3 only)")
    actual: Actual | None = Field(None, description="observed units in the window (observed seasons only)")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "season": "AW2020", "cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
        "style_id": "0751471", "name": "Pluto RW slacks", "raw_name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
        "prediction_score": 1.0, "confidence_top1pct": 1.0, "p_top0_1pct": 1.0, "forecast_units": 7417.3,
        "category": {"product_type": "Trousers", "garment_group": "Trousers", "index_group": "Ladieswear"},
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
    seasons: list[str] = Field(description="season ids available on /seasons")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "status": "ok", "model_version": "regressor-20200923-r94+clf-top1pct-r294+clf-top0.1pct-r135",
        "prediction_cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
        "n_styles": 200, "n_styles_scored": 20318, "seasons": ["AW2020", "SS2020"]}]})


class SeasonInfo(BaseModel):
    id: str
    label: str
    cutoff: date
    forecast_window: ForecastWindow
    observed: bool = Field(description="true when actual units are available (backtest season)")
    has_concepts: bool
    n_styles: int
    default: bool


class SeasonsResponse(BaseModel):
    seasons: list[SeasonInfo]

    model_config = ConfigDict(json_schema_extra={"examples": [{"seasons": [
        {"id": "AW2020", "label": "Autumn/Winter 2020 (forecast)", "cutoff": "2020-09-23",
         "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"}, "observed": False, "has_concepts": True,
         "n_styles": 200, "default": True},
        {"id": "SS2020", "label": "Spring/Summer 2020 (backtest)", "cutoff": "2020-05-27",
         "forecast_window": {"start": "2020-05-27", "end": "2020-06-23"}, "observed": True, "has_concepts": False,
         "n_styles": 200, "default": False}]}]})


class ModelSummary(BaseModel):
    success_definition: str
    horizon: str
    stock_caveat: str
    ranking: str
    regressor: dict[str, Any] = Field(description="validation week and 10-cutoff backtest vs baselines")
    classifier_top1pct: dict[str, Any]
    classifier_top0_1pct: dict[str, Any] = Field(alias="classifier_top0.1pct", serialization_alias="classifier_top0.1pct")
    seasonal: dict[str, Any] = Field(description="category mix of the predicted top-100, SS2020 vs AW2020")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"examples": [{
        "success_definition": "A winner is a style whose units over the next 4 weeks rank in the top 1% …",
        "horizon": "4 weeks: …", "stock_caveat": "There is no stock data, so sales are censored demand …",
        "ranking": "Styles are ranked by the regressor's forecast units; …",
        "regressor": {"backtest": [{"method": "LightGBM regressor", "ndcg@50": {"mean": 0.924, "std": 0.031}}]},
        "classifier_top1pct": {"calibration": {"ece_raw": 0.00361, "ece_calibrated": 0.00282},
                               "reliability_plot_url": "/images/classifier/reliability.png"},
        "classifier_top0.1pct": {"methods": [{"method": "LightGBM classifier", "backtest": {"pr_auc": 0.807}}]},
        "seasonal": {"category_mix": [{"product_group": "Swimwear", "SS2020": 0.38, "AW2020": 0.0,
                                       "change_pts": -38.1}], "ss2020_top10_hits": 7}}]})


class ErrorResponse(BaseModel):
    detail: str
    style_id: str | None = None
    season: str | None = None

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
