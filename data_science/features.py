"""Weekly snapshot feature engineering (leakage-safe, data < cutoff only).

A snapshot at cutoff ``c`` (a Wednesday) has one row per style that sold in the 12 weeks
before ``c``. Features use only weeks with ``week_start < c``; the target is units in
``[c, c + horizon)``. Everything is computed from the weekly style cache (data_science.data),
so a full training set of ~45 cutoffs builds in well under a minute.

Approximations (documented, because the cache is weekly):
- ``buyers_w4`` sums weekly distinct buyers (a customer buying in 2 weeks counts twice).
- ``max_price_ever`` = highest *weekly average* price before the cutoff (not single-transaction max).
- ``weeks_since_launch`` is left-censored at the first data week (2018-09-26).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data_science import data

MOMENTUM_WEEKS: list[int] = [1, 2, 4, 8, 12]
CAT_FEATURES: list[str] = [
    "product_type_name", "product_group_name", "garment_group_name",
    "index_group_name", "colour_group_name", "graphical_appearance_name",
]
NUM_FEATURES: list[str] = (
    [f"units_w{k}" for k in MOMENTUM_WEEKS]
    + ["trend_1_4", "trend_4_12", "buyers_w4", "repeat_rate_w4", "online_share_w4",
       "n_active_articles_w4", "avg_price_w4", "discount", "weeks_since_launch", "is_new",
       "ly_units_h", "week_of_year"]
)
FEATURES: list[str] = NUM_FEATURES + CAT_FEATURES
TARGETS: list[str] = ["y_units", "y_log", "y_buyers"]

TRAIN_FIRST_CUTOFF: date = date(2019, 9, 25)
TRAIN_LAST_CUTOFF: date = config.VALID_CUTOFF - timedelta(weeks=config.HORIZON_WEEKS)  # 2020-07-29
TRAIN_PATH = config.CACHE_DIR / "train.parquet"
VALID_PATH = config.CACHE_DIR / "valid.parquet"

_SW: pd.DataFrame | None = None
_ATTRS: pd.DataFrame | None = None


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (once) the weekly style table and the static categorical attributes."""
    global _SW, _ATTRS
    if _SW is None:
        _SW = data.style_weekly()
    if _ATTRS is None:
        # Article metadata is static (not time-dependent), so the category modes are leakage-safe.
        _ATTRS = data.style_attributes()[["product_code", *CAT_FEATURES]]
    return _SW, _ATTRS


def default_cutoffs(first: date = TRAIN_FIRST_CUTOFF, last: date = TRAIN_LAST_CUTOFF) -> list[date]:
    """Weekly Wednesday cutoffs from ``first`` to ``last`` inclusive."""
    return data.week_grid(first, last + timedelta(weeks=1))


def make_snapshot(cutoff: date, horizon_weeks: int = config.HORIZON_WEEKS,
                  sw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature/target snapshot for one cutoff.

    Rows: styles with any sales in the 12 weeks before ``cutoff``. Columns: product_code,
    cutoff, FEATURES, TARGETS. Targets are NaN when the horizon runs past the end of the
    data (e.g. at FINAL_CUTOFF). ``sw`` overrides the weekly table (used by leakage tests).
    """
    config.assert_cutoff(cutoff)
    if sw is None:
        sw, attrs = _load()
    else:
        attrs = _load()[1]
    c = pd.Timestamp(cutoff)
    ws = sw["week_start"]

    hist = sw[ws < c]  # <- the ONLY frame features are computed from
    assert hist["week_start"].max() < c, "feature leakage: history contains weeks >= cutoff"
    h12 = hist[hist["week_start"] >= c - pd.Timedelta(weeks=12)].copy()
    h12["age"] = (c - h12["week_start"]).dt.days // 7  # 1 = the week just before cutoff, 12 = oldest
    g = h12.groupby("product_code")

    f = pd.DataFrame({f"units_w{k}": h12["units"].where(h12["age"] <= k, 0).groupby(h12["product_code"]).sum()
                      for k in MOMENTUM_WEEKS})
    f = f[f["units_w12"] > 0]
    m4 = h12[h12["age"] <= 4]
    g4 = m4.groupby("product_code")
    f["buyers_w4"] = g4["buyers"].sum()
    rev4 = g4["revenue"].sum()
    online4 = (m4["units"] * m4["online_share"]).groupby(m4["product_code"]).sum()
    f["n_active_articles_w4"] = g4["n_active_articles"].max()
    f = f.fillna({"buyers_w4": 0, "n_active_articles_w4": 0})

    u4 = f["units_w4"].replace(0, np.nan)
    f["trend_1_4"] = (f["units_w1"] / (u4 / 4)).fillna(0)
    f["trend_4_12"] = (f["units_w4"] / 4) / (f["units_w12"] / 12)
    f["repeat_rate_w4"] = (f["units_w4"] / f["buyers_w4"].replace(0, np.nan)).fillna(0)
    f["online_share_w4"] = (online4 / u4).reindex(f.index)
    # Styles with no sales in the last 4 weeks: fall back to 12-week price.
    price12 = g["revenue"].sum() / g["units"].sum()
    f["avg_price_w4"] = (rev4 / u4).reindex(f.index).fillna(price12)
    hg = hist[hist["product_code"].isin(f.index)].groupby("product_code")
    f["discount"] = (1 - f["avg_price_w4"] / hg["avg_price"].max()).clip(lower=0)
    f["weeks_since_launch"] = (c - hg["week_start"].min()).dt.days // 7
    f["is_new"] = (f["weeks_since_launch"] < 8).astype("int8")

    ly0 = c - pd.Timedelta(days=config.LY_OFFSET_DAYS)
    ly = hist[(hist["week_start"] >= ly0) & (hist["week_start"] < ly0 + pd.Timedelta(weeks=horizon_weeks))]
    f["ly_units_h"] = ly.groupby("product_code")["units"].sum().reindex(f.index).fillna(0)
    f["week_of_year"] = int(c.isocalendar().week)

    # --- target: strictly [cutoff, cutoff + horizon) ---
    end = c + pd.Timedelta(weeks=horizon_weeks)
    if end <= pd.Timestamp(config.FINAL_CUTOFF):
        fut = sw[(ws >= c) & (ws < end)].groupby("product_code")[["units", "buyers"]].sum()
        f["y_units"] = fut["units"].reindex(f.index).fillna(0)
        f["y_buyers"] = fut["buyers"].reindex(f.index).fillna(0)
        f["y_log"] = np.log1p(f["y_units"])
    else:
        f[TARGETS] = np.nan

    out = f.reset_index().rename(columns={"index": "product_code"})
    out = out.merge(attrs, on="product_code", how="left")
    out.insert(1, "cutoff", c)
    return out[["product_code", "cutoff", *FEATURES, *TARGETS]]


def build_training_set(cutoffs: list[date] | None = None, path=TRAIN_PATH) -> pd.DataFrame:
    """Concatenate snapshots for ``cutoffs`` (default: weekly 2019-09-25..2020-07-29) and save."""
    cutoffs = cutoffs or default_cutoffs()
    df = pd.concat([make_snapshot(c) for c in cutoffs], ignore_index=True)
    for col in CAT_FEATURES:  # unify categories across snapshots
        df[col] = df[col].astype(str).astype("category")
    if path is not None:
        df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    import time

    t = time.time()
    tr = build_training_set()
    va = make_snapshot(config.VALID_CUTOFF)
    va.to_parquet(VALID_PATH, index=False)
    per = tr.groupby("cutoff").size()
    print(f"train: {len(tr):,} rows, {per.size} cutoffs ({per.index.min():%Y-%m-%d}..{per.index.max():%Y-%m-%d}), "
          f"{per.min():,}-{per.max():,} styles/cutoff | NaN targets: {tr[TARGETS].isna().sum().sum()}")
    print(f"valid: {len(va):,} rows at {config.VALID_CUTOFF} | built in {time.time() - t:.0f}s")
    print(tr[NUM_FEATURES + TARGETS].describe().T[["mean", "50%", "max"]].round(2).to_string())
