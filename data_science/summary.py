"""Write outputs/model_summary.json for the API's /model/summary, from the existing evaluation outputs.

  python -m data_science.summary

Reads (never rewrites) outputs/figures/eval_table.md and seasonal_comparison.md (regressor), and
outputs/classifier/per_cutoff*.csv (classifiers). Every number in the JSON comes from those files.
"""
from __future__ import annotations

import json
import re

import pandas as pd

import config
from data_science import classify

OUT_PATH = config.OUT_DIR / "model_summary.json"
FIG = config.FIGURES_DIR


def _md_rows(text: str, after: str) -> list[list[str]]:
    """Cells of the first markdown table that follows the line containing ``after``."""
    lines = text[text.index(after):].splitlines()
    rows, started = [], False
    for ln in lines[1:]:
        if ln.startswith("|"):
            started = True
            if not set(ln.replace("|", "").strip()) <= set("-: "):
                rows.append([c.strip().strip("*") for c in ln.strip("|").split("|")])
        elif started:
            break
    return rows


def _pm(cell: str) -> dict:
    m = re.match(r"([\d.]+) ± ([\d.]+)", cell)
    return {"mean": float(m[1]), "std": float(m[2])}


def regressor_summary() -> dict:
    text = (FIG / "eval_table.md").read_text(encoding="utf-8")
    valid = _md_rows(text, "### Validation cutoff")
    head, body = valid[0], valid[1:]
    keep = ["precision@12", "ndcg@50", "wape_top500"]
    validation = [{"method": r[0], **{k: (None if r[head.index(k)] == "n/a" else float(r[head.index(k)])) for k in keep}}
                  for r in body]
    bt = _md_rows(text, "### Rolling backtest")
    backtest = [{"method": r[0], "ndcg@50": _pm(r[1]), "precision@12": _pm(r[2])} for r in bt[1:]]
    wins = re.findall(r"- vs (.+?) — ndcg@50: model wins \*\*(\d+)/(\d+)\*\*", text)
    return {"validation_cutoff": str(config.VALID_CUTOFF), "validation": validation,
            "backtest_cutoffs": int(wins[0][2]) if wins else None, "backtest": backtest,
            "ndcg50_wins": [{"baseline": b, "wins": int(w), "of": int(n)} for b, w, n in wins],
            "source": "outputs/figures/eval_table.md"}


def classifier_summary(share: float) -> dict:
    pc = pd.read_csv(classify.paths(share)["per_cutoff"], parse_dates=["cutoff"])
    bt = pc[pc["cutoff"] < pd.Timestamp(config.VALID_CUTOFF)]
    va = pc[pc["cutoff"] == pd.Timestamp(config.VALID_CUTOFF)]
    metrics = classify.RANK_METRICS
    rows = []
    for m in classify.METHODS:
        g, v = bt[bt["method"] == m], va[va["method"] == m]
        rows.append({"method": m, "backtest": {k: round(float(g[k].mean()), 3) for k in metrics},
                     "validation": {k: round(float(v[k].iloc[0]), 3) for k in metrics}})
    clf = pc[pc["method"] == classify.CLF]
    cb, cv = clf[clf["cutoff"] < pd.Timestamp(config.VALID_CUTOFF)], clf[clf["cutoff"] == pd.Timestamp(config.VALID_CUTOFF)]
    piv = bt.pivot(index="cutoff", columns="method", values="pr_auc")
    plot = classify.paths(share)["plot"]
    return {
        "label": f"top {share * 100:g}% of active styles by next-4-week units (per cutoff)",
        "winners_per_cutoff": round(float(clf["n_winners"].mean()), 1),
        "methods": rows,
        "pr_auc_wins_vs_last_week_x4": int(((piv[classify.CLF] - piv[classify.L1]) > 1e-12).sum()),
        "backtest_cutoffs": int(bt["cutoff"].nunique()),
        "calibration": {"ece_raw": round(float(cb["ece_raw"].mean()), 5), "ece_calibrated": round(float(cb["ece_cal"].mean()), 5),
                        "brier_raw": round(float(cb["brier_raw"].mean()), 5), "brier_calibrated": round(float(cb["brier_cal"].mean()), 5),
                        "validation_ece_raw": round(float(cv["ece_raw"].iloc[0]), 5),
                        "validation_ece_calibrated": round(float(cv["ece_cal"].iloc[0]), 5)},
        "reliability_plot": f"outputs/classifier/{plot.name}",
        "source": f"outputs/classifier/{classify.paths(share)['per_cutoff'].name}",
    }


def seasonal_summary() -> dict:
    text = (FIG / "seasonal_comparison.md").read_text(encoding="utf-8")
    mix = _md_rows(text, "#### Category mix")
    head = mix[0]
    rows = [{"product_group": r[0], "SS2020": float(r[1].rstrip("%")) / 100, "AW2020": float(r[2].rstrip("%")) / 100,
             "change_pts": float(r[3])} for r in mix[1:]]
    hits = re.search(r"Backtest: (\d+)/10", text)
    changed = re.search(r"\*\*What changed:\*\* (.+)", text)
    return {"basis": "share of predicted units within each season's top-100 styles, by product group",
            "columns": head, "category_mix": rows,
            "ss2020_top10_hits": int(hits[1]) if hits else None,
            "what_changed": changed[1].strip() if changed else None,
            "source": "outputs/figures/seasonal_comparison.md"}


def build() -> dict:
    return {
        "success_definition": "A winner is a style whose units over the next 4 weeks rank in the top 1% of styles "
                              "active at that cutoff (about 194 of ~19,300; threshold set per cutoff). The displayed "
                              "prediction_score uses the stricter top 0.1% (about 20 styles).",
        "horizon": "Styles turn over fast (median 19 active weeks), so 4 weeks is the longest window that can be "
                   "tested honestly; from the 23 Sep 2020 cutoff it covers the opening of autumn.",
        "stock_caveat": "There is no stock data, so sales are censored demand: a sold-out style looks like a weak "
                        "seller. 'Sold in the last 2 weeks' is the only availability proxy.",
        "ranking": "Styles are ranked by the regressor's forecast units; the classifiers add calibrated probabilities.",
        "regressor": regressor_summary(),
        "classifier_top1pct": classifier_summary(classify.WINNER_SHARE),
        "classifier_top0.1pct": classifier_summary(classify.STRICT_SHARE),
        "seasonal": seasonal_summary(),
    }


def main() -> None:
    out = build()
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote outputs/{OUT_PATH.name}")


if __name__ == "__main__":
    main()
