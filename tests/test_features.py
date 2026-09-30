"""Leakage and sanity tests for forecasting.features."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from forecasting import data, features

CUTOFF = date(2020, 3, 4)


@pytest.fixture(scope="module")
def sw() -> pd.DataFrame:
    """Full weekly style table."""
    return data.style_weekly()


def test_no_feature_leakage(sw: pd.DataFrame) -> None:
    """Features must be identical whether or not data >= cutoff exists (future data is scrambled)."""
    full = features.make_snapshot(CUTOFF, sw=sw)
    scrambled = sw.copy()
    fut = scrambled["week_start"] >= pd.Timestamp(CUTOFF)
    scrambled.loc[fut, ["units", "buyers", "revenue", "avg_price"]] *= 7
    other = features.make_snapshot(CUTOFF, sw=scrambled)
    pd.testing.assert_frame_equal(full[features.FEATURES], other[features.FEATURES])
    assert not full["y_units"].equals(other["y_units"]), "target should depend on future data"


def test_max_feature_date_before_cutoff(sw: pd.DataFrame) -> None:
    """Truncating data at the cutoff reproduces the same features (max feature date < cutoff)."""
    trunc = sw[sw["week_start"] < pd.Timestamp(CUTOFF)]
    assert trunc["week_start"].max() < pd.Timestamp(CUTOFF)
    a = features.make_snapshot(CUTOFF, sw=sw)[features.FEATURES]
    b = features.make_snapshot(CUTOFF, sw=trunc)[features.FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_row_count_and_targets(sw: pd.DataFrame) -> None:
    """One row per style active in the 12 weeks before cutoff; targets complete and consistent."""
    snap = features.make_snapshot(CUTOFF, sw=sw)
    c = pd.Timestamp(CUTOFF)
    active = sw[(sw["week_start"] >= c - pd.Timedelta(weeks=12)) & (sw["week_start"] < c)]["product_code"]
    assert len(snap) == active.nunique() == snap["product_code"].nunique()
    assert 10_000 < len(snap) < 30_000
    assert snap[features.TARGETS].notna().all().all()
    fut = sw[(sw["week_start"] >= c) & (sw["week_start"] < c + pd.Timedelta(weeks=4))]
    expected = fut.groupby("product_code")["units"].sum().reindex(snap["product_code"]).fillna(0)
    np.testing.assert_array_equal(snap["y_units"].to_numpy(), expected.to_numpy())


def test_final_cutoff_has_no_target(sw: pd.DataFrame) -> None:
    """At FINAL_CUTOFF the horizon is beyond the data, so targets are NaN (never zero-filled)."""
    snap = features.make_snapshot(config.FINAL_CUTOFF, sw=sw)
    assert snap["y_units"].isna().all()
    assert len(snap) > 10_000


def test_default_cutoffs() -> None:
    """Train cutoffs are weekly Wednesdays whose targets end by the validation cutoff."""
    cs = features.default_cutoffs()
    assert all(c.weekday() == 2 for c in cs)
    assert cs[-1] + timedelta(weeks=config.HORIZON_WEEKS) == config.VALID_CUTOFF
