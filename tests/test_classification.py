"""Classification and calibration helpers in forecasting.evaluate (hand-worked expected values, no sklearn)."""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.evaluate import (Isotonic, average_precision, brier, classification_metrics, ece,
                                  precision_at_k, recall_at_k, reliability)


def test_average_precision_hand_worked() -> None:
    """Order by score: 1,0,1,0 -> AP = (1/1 + 2/3) / 2 (sklearn docs example gives 0.8333...)."""
    assert average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.8333333333333333)


def test_average_precision_ties_form_one_block() -> None:
    """All scores tied -> one threshold: precision = prevalence, recall jumps 0 -> 1."""
    assert average_precision([1, 0, 0, 0], [0.5] * 4) == pytest.approx(0.25)
    # tie at the top between a positive and a negative: P=1/2 at R=1/2, then P=2/3 at R=1
    assert average_precision([1, 0, 1, 0], [0.9, 0.9, 0.5, 0.1]) == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)


def test_average_precision_edges() -> None:
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))
    assert average_precision([0, 1, 1], [0.0, 0.5, 0.9]) == pytest.approx(1.0)


def test_precision_recall_at_k() -> None:
    y, s = np.array([1, 0, 1, 0, 1]), np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, s, 2) == 0.5
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(np.zeros(5), s, 3) == 0.0
    m = classification_metrics(y, s)
    assert set(m) == {"pr_auc", "precision@20", "precision@50", "recall@50"} and m["recall@50"] == 1.0


def test_brier_reliability_ece() -> None:
    y, p = np.array([0, 1, 1, 0]), np.array([0.0, 1.0, 0.5, 0.5])
    assert brier(y, p) == pytest.approx(0.125)
    r = reliability(y, p)
    assert r["n"].sum() == 4  # p = 1.0 falls in the last bin
    assert ece(y, p) == pytest.approx(0.0)  # each bin: mean predicted == observed
    assert ece(np.array([0, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_isotonic_pava_and_flat_blocks() -> None:
    """y = 1,3,2,4 -> blocks {1}, {2.5 over x=2..3}, {4}; flat inside a block, linear between, clipped outside."""
    iso = Isotonic().fit([1, 2, 3, 4], [1, 3, 2, 4])
    assert iso.predict([1, 2, 2.5, 3, 3.5, 4]).tolist() == pytest.approx([1, 2.5, 2.5, 2.5, 3.25, 4])
    assert iso.predict([0, 9]).tolist() == [1, 4]


def test_isotonic_pools_tied_x_and_is_monotone() -> None:
    iso = Isotonic().fit([1, 1, 2, 2], [0, 1, 1, 1])
    assert iso.predict([1, 1.5, 2]).tolist() == pytest.approx([0.5, 0.75, 1.0])
    rng = np.random.default_rng(0)
    x = rng.random(2000)
    iso = Isotonic().fit(x, (rng.random(2000) < x).astype(float))
    assert np.all(np.diff(iso.x_) > 0) and np.all(np.diff(iso.y_) >= 0)


def test_isotonic_round_trip() -> None:
    iso = Isotonic().fit([1, 2, 3, 4], [1, 3, 2, 4])
    assert Isotonic.from_dict(iso.to_dict()).predict([2.2, 3.7]).tolist() == iso.predict([2.2, 3.7]).tolist()
