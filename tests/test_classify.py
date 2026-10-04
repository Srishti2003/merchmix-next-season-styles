"""Winner label threshold and calibration leakage in forecasting.classify (synthetic data, no Parquet needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import classify


def _snap(cutoff: str, units: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"cutoff": pd.Timestamp(cutoff), "y_units": units})


def test_threshold_is_top_one_percent_per_cutoff() -> None:
    """250 styles -> ceil(2.5) = 3 winners; 100 styles -> 1 winner. Each cutoff uses its own threshold."""
    a = _snap("2020-06-03", list(range(250)))            # top 3: 249, 248, 247
    b = _snap("2020-06-10", [1000 + u for u in range(100)])  # top 1: 1099 (all above cutoff a's values)
    y = classify.winner_labels(pd.concat([a, b], ignore_index=True))
    ya, yb = y[:250], y[250:]
    assert ya.sum() == 3 and set(np.flatnonzero(ya)) == {247, 248, 249}
    assert yb.sum() == 1 and yb[-1] == 1


def test_ties_at_threshold_all_count_and_zero_sales_never_win() -> None:
    """rank(method='min'): styles tied at the threshold share its rank, so all are winners."""
    df = _snap("2020-06-03", [50, 40, 40] + [1] * 97)       # 100 styles -> 1 slot; 50 wins alone
    assert classify.winner_labels(df).sum() == 1
    df = _snap("2020-06-03", [40, 40] + [1] * 98)           # tie for the single slot -> both win
    assert classify.winner_labels(df)[:2].tolist() == [1, 1]
    assert classify.winner_labels(_snap("2020-06-03", [0] * 100)).sum() == 0


def test_calibration_history_uses_only_known_labels() -> None:
    """At cutoff c only OOF rows with cutoff <= c - 4 weeks may be used (their 4-week target has ended)."""
    cuts = pd.date_range("2020-04-01", periods=10, freq="7D")
    oof = pd.DataFrame({"cutoff": np.repeat(cuts, 3), "p_raw": 0.1, "y": 0})
    c = cuts[-1]
    used = classify.history_for(oof, c)["cutoff"].unique()
    assert pd.Timestamp(used.max()) == c - pd.Timedelta(weeks=4)
    assert (used + pd.Timedelta(weeks=4) <= c).all()
    assert set(cuts[-4:]).isdisjoint(used)  # the 3 weeks just before c and c itself are excluded


def test_calibrator_ignores_future_labels() -> None:
    """Changing labels at cutoffs not yet known at c must not change the calibrator used at c."""
    cuts = pd.date_range("2020-04-01", periods=8, freq="7D")
    rng = np.random.default_rng(0)
    oof = pd.DataFrame({"cutoff": np.repeat(cuts, 200), "p_raw": rng.uniform(size=1600)})
    oof["y"] = (rng.uniform(size=1600) < oof["p_raw"]).astype(int)
    c = cuts[-1]
    flipped = oof.copy()
    recent = flipped["cutoff"] > c - pd.Timedelta(weeks=4)
    flipped.loc[recent, "y"] = 1 - flipped.loc[recent, "y"]
    x = np.linspace(0, 1, 50)
    np.testing.assert_array_equal(classify.fit_calibrator(oof, c).predict(x),
                                  classify.fit_calibrator(flipped, c).predict(x))
    assert classify.fit_calibrator(oof, cuts[0]) is None  # nothing known yet at the first cutoff
