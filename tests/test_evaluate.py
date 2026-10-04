"""The pure-numpy ndcg_score must behave like sklearn.metrics.ndcg_score (hard-coded expected values)."""
from __future__ import annotations

import numpy as np
import pytest

from data_science.evaluate import ndcg_score, ranking_metrics


def test_sklearn_docs_example() -> None:
    """sklearn's documented example: DCG 5 + 1/log2(3) + 10/log2(6) over IDCG 10 + 5/log2(3) + 1/log2(4)."""
    assert ndcg_score([[10, 0, 0, 1, 5]], [[.1, .2, .3, 4, 70]]) == pytest.approx(0.6956940443813076, abs=1e-12)


def test_ties_share_average_gain() -> None:
    """Tied top scores (10 and 5) share gain 7.5 at rank 1 -> 7.5 / 10 (sklearn's default tie handling)."""
    assert ndcg_score([[10, 0, 0, 1, 5]], [[1, 0, 0, 0, 1]], k=1) == pytest.approx(0.75, abs=1e-12)


def test_hand_worked_tie_groups() -> None:
    """Three tie groups (avg gains 2.5, 2, 1 at ranks 1-2, 3-4, 5-6) vs ideal order 3,3,2,2,1,0."""
    assert ndcg_score([[3, 2, 3, 0, 1, 2]], [[2, 2, 1, 0, 1, 0]]) == pytest.approx(0.9356871587449666, abs=1e-12)


def test_zero_idcg_row_scores_zero_and_rows_are_averaged() -> None:
    """A query with no relevant items scores 0.0; the result is the mean over rows."""
    assert ndcg_score([[10, 0, 0, 1, 5], [0, 0, 0, 0, 0]], [[1, 0, 0, 0, 1], [1, 2, 3, 4, 5]], k=1) == 0.375


def test_perfect_ranking_and_k_cutoff() -> None:
    """Perfect order -> 1.0 for any k; items beyond k do not count."""
    y = np.array([[5., 4, 3, 2, 1]])
    for k in (None, 1, 3, 5, 50):
        assert ndcg_score(y, y, k=k) == pytest.approx(1.0)
    # only rank 1 counts at k=1: best item (5) placed last -> rank-1 gain is 4 -> 4/5
    assert ndcg_score(y, [[1., 5, 4, 3, 2]], k=1) == pytest.approx(0.8)


def test_input_validation() -> None:
    """1-D input, shape mismatch, single document and negative gains are rejected like sklearn."""
    for yt, ys in (([1, 2, 3], [1, 2, 3]), ([[1, 2]], [[1, 2, 3]]), ([[1]], [[1]]), ([[-1, 2]], [[1, 2]])):
        with pytest.raises(ValueError):
            ndcg_score(yt, ys)


def test_ranking_metrics_uses_2d_ndcg_at_50() -> None:
    """The only call site: ranking_metrics passes one query row with k=50."""
    y = np.arange(100, dtype=float)
    assert ranking_metrics(y, y)["ndcg@50"] == pytest.approx(1.0)
