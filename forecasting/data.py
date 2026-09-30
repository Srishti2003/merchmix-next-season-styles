"""DuckDB data layer over the transaction and article Parquet files.

Conventions (see config.py): weeks run Wednesday -> Tuesday and are identified by
their Wednesday ``week_start``. A style is a ``product_code`` (= article_id // 1000).
Transactions before ``FIRST_WEEK`` (a partial week) are dropped from weekly data.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

import config

STYLE_WEEKLY_CACHE: Path = config.CACHE_DIR / "style_weekly.parquet"
LAST_WEEK: date = config.FINAL_CUTOFF - timedelta(days=7)  # 2020-09-16 (16-22 Sep)

CAT_COLS: list[str] = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_value_name", "department_name",
    "index_name", "index_group_name", "section_name", "garment_group_name",
]

# Python weekday: Mon=0 ... Wed=2. DuckDB isodow: Mon=1 ... Wed=3.
# (weekday - 2) mod 7 == (isodow - 3) mod 7 == (isodow + 4) mod 7
_WEEK_START_SQL = "t_dat - CAST((isodow(t_dat) + 4) % 7 AS INTEGER)"
_PRODUCT_CODE_SQL = "lpad(CAST(article_id // 1000 AS VARCHAR), 7, '0')"


@lru_cache(maxsize=1)
def get_con() -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB connection (2 GB memory cap, spills to outputs/cache)."""
    config.ensure_dirs()
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_progress_bar = false")  # never write to stdout (breaks MCP stdio)
    con.execute(f"SET temp_directory = '{(config.CACHE_DIR / 'duckdb_tmp').as_posix()}'")
    return con


def week_start_of(d: date) -> date:
    """Return the Wednesday that starts the Wed->Tue week containing ``d``."""
    return d - timedelta(days=(d.weekday() - 2) % 7)


def week_grid(start: date = config.FIRST_WEEK, end: date = config.FINAL_CUTOFF) -> list[date]:
    """Return all week_start Wednesdays in the half-open range [start, end)."""
    config.assert_cutoff(start)
    config.assert_cutoff(end)
    n = (end - start).days // 7
    return [start + timedelta(weeks=i) for i in range(n)]


def season_of(d: date) -> str:
    """Meteorological season (northern hemisphere) of the date's month."""
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(d.month, "autumn")


def season_label(d: date) -> str:
    """Season plus season-year, e.g. 'winter 2019/20' for Jan 2020, 'summer 2020'."""
    s = season_of(d)
    if s == "winter":
        y0 = d.year if d.month == 12 else d.year - 1
        return f"winter {y0}/{str(y0 + 1)[2:]}"
    return f"{s} {d.year}"


def _check_full_weeks(con: duckdb.DuckDBPyConnection) -> None:
    """Assert every week in [FIRST_WEEK, FINAL_CUTOFF) has exactly 7 distinct days of data."""
    bad = con.execute(f"""
        SELECT {_WEEK_START_SQL} AS week_start, count(DISTINCT t_dat) AS n_days
        FROM read_parquet('{config.TX_GLOB}')
        WHERE t_dat >= ? AND t_dat < ?
        GROUP BY 1 HAVING count(DISTINCT t_dat) <> 7
    """, [config.FIRST_WEEK, config.FINAL_CUTOFF]).fetchall()
    assert not bad, f"Weeks without 7 days of data: {bad}"


def build_style_weekly() -> Path:
    """Aggregate transactions to (product_code, week_start) and write the Parquet cache."""
    con = get_con()
    _check_full_weeks(con)
    con.execute(f"""
        COPY (
            SELECT {_PRODUCT_CODE_SQL} AS product_code,
                   {_WEEK_START_SQL}   AS week_start,
                   count(*)                          AS units,
                   count(DISTINCT cust)              AS buyers,
                   sum(price)                        AS revenue,
                   avg(price)                        AS avg_price,
                   avg(CAST(channel = 2 AS DOUBLE))  AS online_share,
                   count(DISTINCT article_id)        AS n_active_articles
            FROM read_parquet('{config.TX_GLOB}')
            WHERE t_dat >= DATE '{config.FIRST_WEEK}' AND t_dat < DATE '{config.FINAL_CUTOFF}'
            GROUP BY ALL
        ) TO '{STYLE_WEEKLY_CACHE.as_posix()}' (FORMAT parquet)
    """)
    return STYLE_WEEKLY_CACHE


def style_weekly(start: date | None = None, end: date | None = None,
                 rebuild: bool = False) -> pd.DataFrame:
    """Weekly style sales: one row per (product_code, week_start) with sales that week.

    Columns: product_code, week_start, units, buyers, revenue, avg_price, online_share,
    n_active_articles. ``start``/``end`` filter week_start to [start, end) and must be
    Wednesdays. Weeks with zero sales are absent (not zero-filled). Built once and cached.
    """
    if rebuild or not STYLE_WEEKLY_CACHE.exists():
        build_style_weekly()
    start = config.assert_cutoff(start) if start else config.FIRST_WEEK
    end = config.assert_cutoff(end) if end else config.FINAL_CUTOFF
    df = get_con().execute(f"""
        SELECT * FROM read_parquet('{STYLE_WEEKLY_CACHE.as_posix()}')
        WHERE week_start >= ? AND week_start < ?
        ORDER BY product_code, week_start
    """, [start, end]).df()
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def style_attributes(as_of: date | None = None) -> pd.DataFrame:
    """One row per product_code: attribute modes, colours, launch, price and best-seller article.

    Sales-derived columns (first_sale_date, max_price_ever, units_total,
    representative_article_id) use only transactions with t_dat < ``as_of``
    (default: all data), so the result is leakage-safe for a given cutoff.
    Styles with no sales before ``as_of`` have null first_sale_date.
    """
    as_of = config.assert_cutoff(as_of) if as_of else config.FINAL_CUTOFF
    df = get_con().execute(f"""
        WITH sales AS (
            SELECT article_id, count(*) AS units, min(t_dat) AS first_sale, max(price) AS max_price
            FROM read_parquet('{config.TX_GLOB}')
            WHERE t_dat < ?
            GROUP BY article_id
        ),
        art AS (
            SELECT a.*, coalesce(s.units, 0) AS units, s.first_sale, s.max_price,
                   row_number() OVER (PARTITION BY a.product_code
                                      ORDER BY coalesce(s.units, 0) DESC, a.article_id) AS rk
            FROM read_parquet('{config.ARTICLES_PATH}') a
            LEFT JOIN sales s USING (article_id)
        )
        SELECT product_code,
               count(*)                          AS n_articles,
               count(DISTINCT colour_group_name) AS n_colours,
               min(first_sale)                   AS first_sale_date,
               max(max_price)                    AS max_price_ever,
               sum(units)                        AS units_total,
               arg_min(article_id, rk)           AS representative_article_id,
               arg_min(prod_name, rk)            AS prod_name,
               arg_min(detail_desc, rk)          AS detail_desc
        FROM art
        GROUP BY product_code
        ORDER BY product_code
    """, [as_of]).df()
    df = _attribute_modes().merge(df, on="product_code", how="right")
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


@lru_cache(maxsize=1)
def _attribute_modes() -> pd.DataFrame:
    """Most common value of each categorical attribute per style (ties -> alphabetical, deterministic)."""
    art = get_con().execute(
        f"SELECT product_code, {', '.join(CAT_COLS)} FROM read_parquet('{config.ARTICLES_PATH}')").df()
    out = pd.DataFrame({"product_code": sorted(art["product_code"].unique())})
    for c in CAT_COLS:
        m = (art.groupby(["product_code", c]).size().rename("n").reset_index()
             .sort_values(["product_code", "n", c], ascending=[True, False, True])
             .drop_duplicates("product_code")[["product_code", c]])
        out = out.merge(m, on="product_code", how="left")
    return out


def sales_curve(product_code: str, weeks: int | None = None,
                end: date = config.FINAL_CUTOFF) -> pd.DataFrame:
    """Zero-filled weekly units and buyers for one style, from its first sale week to ``end``.

    If ``weeks`` is given, return only the last ``weeks`` weeks before ``end``.
    Returns an empty frame if the style has no sales before ``end``.
    """
    config.assert_cutoff(end)
    code = str(product_code).zfill(7)
    if not STYLE_WEEKLY_CACHE.exists():
        build_style_weekly()
    df = get_con().execute(f"""
        SELECT week_start, units, buyers FROM read_parquet('{STYLE_WEEKLY_CACHE.as_posix()}')
        WHERE product_code = ? AND week_start < ? ORDER BY week_start
    """, [code, end]).df()
    if df.empty:
        return pd.DataFrame(columns=["week_start", "units", "buyers"])
    first = pd.Timestamp(df["week_start"].min()).date()
    start = first if weeks is None else max(first, end - timedelta(weeks=weeks))
    grid = pd.DatetimeIndex(week_grid(start, end), name="week_start")
    df["week_start"] = pd.to_datetime(df["week_start"])
    out = df.set_index("week_start").reindex(grid, fill_value=0).reset_index()
    return out.astype({"units": "int64", "buyers": "int64"})


def top_colour_articles(product_code: str, n: int = 3, as_of: date = config.FINAL_CUTOFF,
                        weeks: int = 12) -> pd.DataFrame:
    """Best-selling article per colour for one style, top ``n`` colours by units.

    Uses transactions in the ``weeks`` weeks before ``as_of`` (falls back to all history if the
    style had no sales then). Columns: article_id, colour_group_name, units, image_path
    (Kaggle path ``images/{aid10[:3]}/{aid10}.jpg``).
    """
    config.assert_cutoff(as_of)
    code = str(product_code).zfill(7)
    lo = as_of - timedelta(weeks=weeks)
    q = f"""
        WITH s AS (
            SELECT article_id, count(*) AS units FROM read_parquet('{config.TX_GLOB}')
            WHERE article_id // 1000 = ? AND t_dat >= ? AND t_dat < ? GROUP BY article_id
        ), a AS (
            SELECT a.article_id, a.colour_group_name, coalesce(s.units, 0) AS units
            FROM read_parquet('{config.ARTICLES_PATH}') a LEFT JOIN s USING (article_id)
            WHERE a.product_code = ?
        )
        SELECT arg_max(article_id, units) AS article_id, colour_group_name, max(units) AS units,
               sum(units) AS colour_units
        FROM a GROUP BY colour_group_name ORDER BY colour_units DESC, article_id LIMIT ?
    """
    df = get_con().execute(q, [int(code), lo, as_of, code, n]).df()
    if df["units"].sum() == 0 and lo > config.FIRST_WEEK:
        return top_colour_articles(code, n, as_of, weeks=10_000)
    aid10 = df["article_id"].astype(str).str.zfill(10)
    df["image_path"] = "images/" + aid10.str[:3] + "/" + aid10 + ".jpg"
    return df[["article_id", "colour_group_name", "units", "image_path"]]


SEASONS: dict[str, tuple[int, ...]] = {"spring": (3, 4, 5), "summer": (6, 7, 8),
                                        "autumn": (9, 10, 11), "winter": (12, 1, 2)}


def season_summary(season: str, year: int, n: int = 10) -> dict:
    """Actual sales summary for one season: category mix and top-``n`` styles by units.

    Weeks are assigned to the season of their middle day (Saturday). ``year`` is the year the
    season starts in, so winter 2019 = Dec 2019 – Feb 2020. Raises ValueError if no data.
    """
    if season not in SEASONS:
        raise ValueError(f"season must be one of {list(SEASONS)}, got {season!r}")
    sw = style_weekly()
    mid = sw["week_start"] + pd.Timedelta(days=3)
    s_year = mid.dt.year - ((season == "winter") & (mid.dt.month <= 2)).astype(int)
    df = sw[mid.dt.month.isin(SEASONS[season]) & (s_year == year)]
    if df.empty:
        raise ValueError(f"No data for {season} {year}. Data covers autumn 2018 (from 26 Sep) to autumn 2020 "
                         "(to 22 Sep); winter uses its starting year (winter 2019 = Dec 2019–Feb 2020).")
    attrs = style_attributes()[["product_code", "prod_name", "product_type_name", "product_group_name",
                                "garment_group_name"]]
    df = df.merge(attrs, on="product_code", how="left")
    total = int(df["units"].sum())
    n_weeks = int(df["week_start"].nunique())

    def shares(col: str, k: int) -> list[dict]:
        s = df.groupby(col, observed=True)["units"].sum().nlargest(k)
        return [{"name": str(i), "units": int(v), "share": round(v / total, 4)} for i, v in s.items()]

    top = (df.groupby(["product_code", "prod_name", "product_type_name", "garment_group_name"], observed=True)
           ["units"].sum().nlargest(n).reset_index())
    return {
        "season": season, "year": year, "weeks": n_weeks, "partial": n_weeks < 13,
        "date_from": str(df["week_start"].min().date()),
        "date_to": str((df["week_start"].max() + pd.Timedelta(days=6)).date()),
        "total_units": total,
        "top_product_groups": shares("product_group_name", 8),
        "top_product_types": shares("product_type_name", 10),
        "top_styles": [{"product_code": r.product_code, "prod_name": r.prod_name,
                        "product_type_name": str(r.product_type_name),
                        "garment_group_name": str(r.garment_group_name), "units": int(r.units)}
                       for r in top.itertuples()],
    }
