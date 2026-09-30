"""MCP server 'retail': style attributes, sales curves, reference images, season summaries.

Thin FastMCP stdio wrapper over forecasting.data / forecasting.select — no business logic here.
Run: python mcp_servers/retail_data.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from forecasting import data, select  # noqa: E402

mcp = FastMCP("retail", instructions=(
    "H&M sales data 2018-09-20 → 2020-09-22, aggregated to styles (product_code = 7-digit article_id prefix). "
    "Weeks run Wednesday→Tuesday. There is NO stock/availability data."))

ProductCode = Annotated[str, Field(description="7-digit style code, e.g. '0751471' (leading zeros optional)",
                                   pattern=r"^\d{6,7}$")]


class StyleAttributes(BaseModel):
    """Descriptive attributes of a style (most common value across its colour/size articles)."""
    product_code: str
    prod_name: str
    product_type_name: str
    product_group_name: str
    garment_group_name: str
    index_group_name: str
    section_name: str
    department_name: str
    colour_group_name: str = Field(description="most common colour among the style's articles")
    graphical_appearance_name: str
    n_articles: int = Field(description="number of colour variants (article_ids)")
    n_colours: int
    first_sale_date: str | None
    units_total: int = Field(description="units sold 2018-09-20 → 2020-09-22")
    representative_article_id: int = Field(description="best-selling article of the style")
    detail_desc: str | None


class WeekSales(BaseModel):
    """Units and buyers in one Wed→Tue week."""
    week_start: str
    units: int
    buyers: int


class CategoryShare(BaseModel):
    name: str
    units: int
    share: float = Field(description="share of the season's units, 0-1")


class StyleSales(BaseModel):
    product_code: str
    prod_name: str
    product_type_name: str
    garment_group_name: str
    units: int


class SeasonSummary(BaseModel):
    """Actual sales in one season: category mix and best-selling styles."""
    season: str
    year: int
    weeks: int
    partial: bool = Field(description="true if fewer than 13 weeks of data")
    date_from: str
    date_to: str
    total_units: int
    top_product_groups: list[CategoryShare]
    top_product_types: list[CategoryShare]
    top_styles: list[StyleSales]


@lru_cache(maxsize=1)
def _attrs() -> pd.DataFrame:
    """All style attributes (cached for the server's lifetime)."""
    return data.style_attributes().set_index("product_code")


def _code(product_code: str) -> str:
    """Normalise and validate a style code."""
    code = product_code.zfill(7)
    if code not in _attrs().index:
        raise ToolError(f"Unknown product_code {product_code!r}. Use a 7-digit style code such as '0751471' "
                        "(get candidates from forecast.predict_top_k or retail.season_summary).")
    return code


@mcp.tool
def get_style_attributes(product_code: ProductCode) -> StyleAttributes:
    """Get a style's product attributes (type, garment group, colour, pattern, description) and lifetime units.

    Use this to understand WHAT a style is before describing or redesigning it.
    """
    code = _code(product_code)
    r = _attrs().loc[code]
    return StyleAttributes(
        product_code=code, **{k: str(r[k]) for k in (
            "prod_name", "product_type_name", "product_group_name", "garment_group_name", "index_group_name",
            "section_name", "department_name", "colour_group_name", "graphical_appearance_name")},
        n_articles=int(r["n_articles"]), n_colours=int(r["n_colours"]),
        first_sale_date=None if pd.isna(r["first_sale_date"]) else str(pd.Timestamp(r["first_sale_date"]).date()),
        units_total=int(r["units_total"]), representative_article_id=int(r["representative_article_id"]),
        detail_desc=None if pd.isna(r["detail_desc"]) else str(r["detail_desc"]))


@mcp.tool
def get_sales_curve(product_code: ProductCode,
                    weeks: Annotated[int, Field(ge=1, le=52, description="number of most recent weeks")] = 26
                    ) -> list[WeekSales]:
    """Get a style's weekly units and buyers for the last `weeks` weeks of data (zero-filled, oldest first).

    Use this as sales evidence: momentum, peaks, and whether the style is still selling.
    """
    sc = data.sales_curve(_code(product_code), weeks=weeks)
    return [WeekSales(week_start=str(r.week_start.date()), units=int(r.units), buyers=int(r.buyers))
            for r in sc.itertuples()]


@mcp.tool
def get_reference_images(product_code: ProductCode) -> list[str]:
    """List local file paths of the style's reference product photos (best-selling colours first by filename).

    Images exist only for styles that won a forecast run (downloaded into outputs/refs/<code>/).
    Open them with a file-reading tool to see the garment.
    """
    code = _code(product_code)
    paths = select.reference_images(code)
    if not paths:
        raise ToolError(f"No reference images downloaded for {code}. They are fetched for the top-3 winners by "
                        "`python -m forecasting.select` (see outputs/refs/).")
    return [str(p) for p in paths]


@mcp.tool
def season_summary(season: Literal["spring", "summer", "autumn", "winter"],
                   year: Annotated[int, Field(ge=2018, le=2020,
                                              description="year the season starts (winter 2019 = Dec 2019–Feb 2020)")]
                   ) -> SeasonSummary:
    """Summarise ACTUAL sales in a past season: product-group and product-type mix, plus the 10 best-selling styles.

    Data covers autumn 2018 (partial) to autumn 2020 (partial, 3 weeks).
    """
    try:
        return SeasonSummary(**data.season_summary(season, year))
    except ValueError as e:
        raise ToolError(str(e)) from e


if __name__ == "__main__":
    mcp.run(show_banner=False)
