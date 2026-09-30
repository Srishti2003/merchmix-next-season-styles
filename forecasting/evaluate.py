"""Baselines and ranking metrics (precision@k, NDCG, Spearman, WAPE).

All metrics compare a score (higher = stronger) against actual next-4-week units for the
styles in one snapshot. Definitions:
- precision@k: share of the predicted top-k that are in the actual top-k.
- top-12 hit rate in top-50: share of the actual top-12 that appear in the predicted top-50
               ("did the shortlist a buyer would review catch the real hits?").
- NDCG@50:     graded by actual units.
- spearman_top500: rank correlation of score vs actual among the actual top-500.
- wape_top500: sum|pred - actual| / sum actual over the actual top-500 (needs unit forecasts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HIT_RATE = "top-12 hit rate in top-50"

BASELINES: dict[str, str] = {
    "last_4w": "Last 4 weeks units",
    "same_period_ly": "Same 4 weeks last year",
    "last_1w_x4": "Last week × 4",
}


def baseline_forecasts(snap: pd.DataFrame) -> dict[str, pd.Series]:
    """Unit forecasts for the 3 naive baselines, indexed like ``snap``."""
    return {
        "last_4w": snap["units_w4"].astype(float),
        "same_period_ly": snap["ly_units_h"].astype(float),
        "last_1w_x4": snap["units_w1"].astype(float) * 4,
    }


def _tie_averaged_dcg(y_true: np.ndarray, y_score: np.ndarray, discount_cumsum: np.ndarray) -> float:
    """DCG of one row where tied scores share the average gain of their group (sklearn's default)."""
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return float((ranked * discount_sums).sum())


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    """Pure-numpy NDCG@k, same behaviour as ``sklearn.metrics.ndcg_score`` (default arguments).

    2-D inputs, one row per query; linear gains (the y_true values), log2 discount 1/log2(rank+1),
    discounts beyond ``k`` set to 0; tied scores get the average gain of their tie group; the ideal
    DCG ignores ties; a row whose ideal DCG is 0 scores 0.0; the result is the mean over rows.
    (Replaces sklearn, whose compiled DLLs are blocked by a device Application Control policy.)
    """
    y_true, y_score = np.asarray(y_true, dtype=float), np.asarray(y_score, dtype=float)
    if y_true.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError(f"expected 2-D arrays of equal shape, got {y_true.shape} and {y_score.shape}")
    if y_true.shape[1] <= 1:
        raise ValueError("NDCG needs more than one document per query")
    if (y_true < 0).any():
        raise ValueError("y_true must be non-negative")
    discount = 1 / np.log2(np.arange(y_true.shape[1]) + 2)
    if k is not None:
        discount[k:] = 0
    discount_cumsum = np.cumsum(discount)
    scores = []
    for t, s in zip(y_true, y_score):
        ideal = float(np.sort(t)[::-1] @ discount)
        scores.append(0.0 if ideal == 0 else _tie_averaged_dcg(t, s, discount_cumsum) / ideal)
    return float(np.mean(scores))


def _top(s: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values (ties broken by position, deterministic)."""
    return np.argsort(-s, kind="stable")[:k]


def ranking_metrics(y_true: np.ndarray, score: np.ndarray, units_pred: np.ndarray | None = None) -> dict[str, float]:
    """Compute all metrics for one snapshot. ``units_pred`` enables WAPE (else NaN)."""
    y_true, score = np.asarray(y_true, float), np.asarray(score, float)
    t3, t12, t50, t500 = (_top(y_true, k) for k in (3, 12, 50, 500))
    p3, p12, p50 = (_top(score, k) for k in (3, 12, 50))
    out = {
        "precision@3": len(set(p3) & set(t3)) / 3,
        "precision@12": len(set(p12) & set(t12)) / 12,
        HIT_RATE: len(set(p50) & set(t12)) / 12,
        "ndcg@50": float(ndcg_score(y_true[None, :], score[None, :], k=50)),
        "spearman_top500": float(spearmanr(score[t500], y_true[t500]).statistic),
        "wape_top500": np.nan,
    }
    if units_pred is not None:
        u = np.asarray(units_pred, float)
        out["wape_top500"] = float(np.abs(u[t500] - y_true[t500]).sum() / y_true[t500].sum())
    return out


def compare(snap: pd.DataFrame, model_scores: dict[str, tuple[np.ndarray, np.ndarray | None]]) -> pd.DataFrame:
    """Metrics table (rows = methods) for model scores plus the 3 baselines on one snapshot.

    ``model_scores`` maps a method name to (score, unit forecast or None).
    """
    y = snap["y_units"].to_numpy()
    rows = {name: ranking_metrics(y, s, u) for name, (s, u) in model_scores.items()}
    for name, f in baseline_forecasts(snap).items():
        rows[BASELINES[name]] = ranking_metrics(y, f.to_numpy(), f.to_numpy())
    return pd.DataFrame(rows).T


def to_markdown(table: pd.DataFrame, title: str) -> str:
    """Render a metrics table as markdown (best value per column in bold; WAPE lower = better)."""
    cols = list(table.columns)
    lines = [f"### {title}", "", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    best = {c: (table[c].min() if c.startswith("wape") else table[c].max()) for c in cols}
    for name, r in table.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            s = "n/a" if pd.isna(v) else f"{v:.3f}"
            cells.append(f"**{s}**" if not pd.isna(v) and np.isclose(v, best[c]) else s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def movers(snap: pd.DataFrame, score: np.ndarray, n: int = 10, pool: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Where the model disagrees most with the "last week × 4" baseline, and who was right.

    Only styles in the top-``pool`` of either ranking are considered (disagreements deep in the tail
    don't matter to a buyer). Returns (promoted, demoted, 2 summary lines). ``promoted`` = the model
    ranks the style far ABOVE the baseline; a disagreement is "right" when the actual rank is closer
    to the model's rank than to the baseline's (compared on log-rank, so 5 vs 50 counts like 50 vs 500).
    """
    d = snap[["product_code", "units_w1", "y_units"]].copy()
    d["model_rank"] = pd.Series(-np.asarray(score)).rank(method="first").astype(int).to_numpy()
    d["baseline_rank"] = (-4 * d["units_w1"]).rank(method="min").astype(int)
    d["actual_rank"] = (-d["y_units"]).rank(method="min").astype(int)
    d = d[(d["model_rank"] <= pool) | (d["baseline_rank"] <= pool)].copy()
    d["rank_gap"] = d["baseline_rank"] - d["model_rank"]  # > 0: model ranks it higher
    la, lm, lb = (np.log(d[c]) for c in ("actual_rank", "model_rank", "baseline_rank"))
    d["model_right"] = (la - lm).abs() < (la - lb).abs()
    promoted = d.nlargest(n, "rank_gap").reset_index(drop=True)
    demoted = d.nsmallest(n, "rank_gap").reset_index(drop=True)
    lines = []
    for name, g, verb in (("promoted", promoted, "above"), ("demoted", demoted, "below")):
        lines.append(f"Styles the model ranked far {verb} the baseline ({name}): the model was closer to the actual "
                     f"rank for {int(g['model_right'].sum())}/{len(g)} (median actual rank {g['actual_rank'].median():.0f}; "
                     f"median model rank {g['model_rank'].median():.0f} vs baseline rank {g['baseline_rank'].median():.0f}).")
    return promoted, demoted, lines
