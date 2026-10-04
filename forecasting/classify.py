"""Winner classifier: P(style is a top-1% seller over the next 4 weeks), with isotonic calibration.

Label (per cutoff, never global): a style is a winner when its next-4-week units rank in the top 1% of the
styles active at that cutoff, i.e. ``rank(y_units, method="min", descending) <= ceil(0.01 * n_styles)``.

Model: LightGBM binary on ``features.FEATURES`` (same parameters, seed and determinism flags as model.py).
Rounds come from early stopping on the last 4 training cutoffs (metric: average precision), like the regressor.

Evaluation: the regressor's 10-cutoff rolling backtest (2020-05-27 .. 2020-07-29) plus the validation cutoff
2020-08-26. At cutoff c every model is trained only on cutoffs whose target window ended by c
(cutoff <= c - 4 weeks). Baselines are ranked by their score: last week × 4, last 4 weeks units, and the
existing regressor's unit forecast (retrained per fold with its published 94 rounds).

Calibration: the isotonic map used at cutoff c is fitted only on out-of-fold (OOF) predictions whose labels
were known at c (OOF cutoff <= c - 4 weeks). To give the first backtest folds a calibrator, OOF predictions are
also made at N_WARMUP cutoffs before the backtest window; those are used for calibration only, not scored.
The final calibrator uses every OOF row with cutoff <= 2020-08-26 (targets end by 2020-09-22).
Ranking always uses the raw probability (isotonic maps create ties); ``prediction_score`` is the calibrated one.

Two labels are trained: top 1% (default) and a stricter top 0.1% (``--share 0.001``). Each run writes
outputs/classifier/{eval_classifier, per_cutoff, reliability, oof_predictions, scores_clf}<suffix> and
outputs/models/{lgbm_clf<suffix>_final_20200923.txt, isotonic_clf<suffix>_20200923.json, clf<suffix>_meta.json}
(suffix '' for top 1%, '_top0.1pct' for top 0.1%). ``--combine`` then keeps the regressor's ranking (published
top-3), picks prediction_score (P(top 0.1%) if usable, else P(top 1%)), keeps P(top 1%) as confidence_top1pct, and
writes outputs/classifier/{scores_final.parquet, final_scores.md}. The regressor's outputs are never touched.

  python -m forecasting.classify && python -m forecasting.classify --share 0.001 && python -m forecasting.classify --combine
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from forecasting import evaluate, features, model

WINNER_SHARE = 0.01
STRICT_SHARE = 0.001
N_BACKTEST = model.N_BACKTEST
N_WARMUP = 8
OUT_DIR = config.OUT_DIR / "classifier"
PUBLISHED_TOP3 = ["0751471", "0762846", "0685814"]  # regressor top-3 (Pluto, Lucy, RICHIE)

PARAMS = {**model._BASE, "objective": "binary", "metric": "average_precision"}
HORIZON = pd.Timedelta(weeks=config.HORIZON_WEEKS)

CLF = "LightGBM classifier"
REG = "LightGBM regressor (units)"
L1 = evaluate.BASELINES["last_1w_x4"]
L4 = evaluate.BASELINES["last_4w"]
METHODS = [CLF, L1, L4, REG]
RANK_METRICS = ["pr_auc", "precision@20", "precision@50", "recall@50"]


def winner_labels(df: pd.DataFrame, share: float = WINNER_SHARE) -> np.ndarray:
    """1 if y_units is in the top ``share`` of styles at the same cutoff (ties at the threshold all count)."""
    g = df.groupby("cutoff")["y_units"]
    n = g.transform("size").to_numpy()
    rank = g.rank(method="min", ascending=False).to_numpy()
    return ((rank <= np.ceil(share * n)) & (df["y_units"].to_numpy() > 0)).astype(int)


def _dataset(df: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(model.prepare(df), df["y"].to_numpy(), reference=reference, free_raw_data=False)


def best_rounds(train: pd.DataFrame) -> int:
    """Early-stopping round count: train on all but the last model.N_INNER cutoffs, validate on those."""
    cut = sorted(train["cutoff"].unique())[-model.N_INNER]
    tr, va = train[train["cutoff"] < cut], train[train["cutoff"] >= cut]
    dtr = _dataset(tr)
    inner = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[_dataset(va, reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return inner.best_iteration


def fit(train: pd.DataFrame, rounds: int) -> lgb.Booster:
    return lgb.train(PARAMS, _dataset(train), num_boost_round=rounds)


def predict_proba(booster: lgb.Booster, snap: pd.DataFrame) -> np.ndarray:
    return booster.predict(model.prepare(snap))


def history_for(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows whose target window ended by ``cutoff`` (cutoff_row <= cutoff - horizon): the only rows whose
    labels are known at ``cutoff``. Used both for training sets and for calibration data."""
    return df[df["cutoff"] <= pd.Timestamp(cutoff) - HORIZON]


def fit_calibrator(oof: pd.DataFrame, cutoff: pd.Timestamp) -> evaluate.Isotonic | None:
    """Isotonic map from raw OOF probability to winner rate, fitted only on labels known at ``cutoff``."""
    hist = history_for(oof, cutoff)
    return evaluate.Isotonic().fit(hist["p_raw"], hist["y"]) if len(hist) else None


def out_of_fold(all_df: pd.DataFrame, cutoffs: list[pd.Timestamp], rounds: int,
                reg_rounds: int | None) -> pd.DataFrame:
    """Raw classifier probabilities (and, if ``reg_rounds``, regressor unit forecasts) at each cutoff, each
    from models trained only on rows whose targets ended by that cutoff."""
    parts = []
    for c in cutoffs:
        t = time.time()
        tr = history_for(all_df, c)
        snap = all_df[all_df["cutoff"] == c].reset_index(drop=True)
        out = snap[["product_code", "cutoff", "y", "y_units", "units_w1", "units_w2", "units_w4"]].copy()
        out["p_raw"] = predict_proba(fit(tr, rounds), snap)
        if reg_rounds:
            reg = lgb.train(model.PARAMS["regressor"], model._dataset(tr, "regressor"), num_boost_round=reg_rounds)
            out["reg_units"] = model.predict(reg, "regressor", snap)[1]
        parts.append(out)
        print(f"  OOF {c:%Y-%m-%d}: trained on {tr['cutoff'].nunique()} cutoffs ({len(tr):,} rows), "
              f"{len(snap):,} styles, {int(snap['y'].sum())} winners, {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True)


def per_cutoff_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (cutoff, method) with ranking metrics, plus Brier/ECE for the classifier."""
    rows = []
    for c, g in scored.groupby("cutoff", sort=True):
        y = g["y"].to_numpy()
        base = {"cutoff": c, "n_styles": len(g), "n_winners": int(y.sum()), "base_rate": float(y.mean()),
                "threshold_units": float(g.loc[g["y"] == 1, "y_units"].min())}
        scores = {CLF: g["p_raw"], L1: 4 * g["units_w1"], L4: g["units_w4"], REG: g["reg_units"]}
        for name, s in scores.items():
            r = {**base, "method": name, **evaluate.classification_metrics(y, s.to_numpy(float))}
            if name == CLF:
                r |= {"brier_raw": evaluate.brier(y, g["p_raw"]), "ece_raw": evaluate.ece(y, g["p_raw"]),
                      "brier_cal": evaluate.brier(y, g["p_cal"]), "ece_cal": evaluate.ece(y, g["p_cal"])}
            rows.append(r)
    return pd.DataFrame(rows)


def _summary_table(pc: pd.DataFrame) -> list[str]:
    n = pc["cutoff"].nunique()
    lines = ["| method | " + " | ".join(RANK_METRICS) + " |", "|---" * (len(RANK_METRICS) + 1) + "|"]
    for m in METHODS:
        g = pc[pc["method"] == m]
        lines.append(f"| {m} | " + " | ".join(f"{g[k].mean():.3f} ± {g[k].std():.3f}" for k in RANK_METRICS) + " |")
    lines.append("")
    for base in METHODS[1:]:
        parts = []
        for k in RANK_METRICS:
            piv = pc.pivot(index="cutoff", columns="method", values=k)
            d = piv[CLF] - piv[base]
            parts.append(f"{k} {(d > 1e-12).sum()}/{n} (ties {(d.abs() <= 1e-12).sum()})")
        lines.append(f"- classifier wins vs {base}: " + "; ".join(parts))
    return lines


def _valid_table(pc: pd.DataFrame) -> list[str]:
    lines = ["| method | " + " | ".join(RANK_METRICS) + " |", "|---" * (len(RANK_METRICS) + 1) + "|"]
    for m in METHODS:
        r = pc[pc["method"] == m].iloc[0]
        lines.append(f"| {m} | " + " | ".join(f"{r[k]:.3f}" for k in RANK_METRICS) + " |")
    return lines


def reliability_plot(scored: pd.DataFrame, path) -> pd.DataFrame:
    """Pooled reliability curve (raw vs calibrated) over the scored cutoffs; returns the two tables."""
    import matplotlib.pyplot as plt

    from forecasting import viz

    viz.apply_style()
    y = scored["y"].to_numpy()
    tabs = {"raw": evaluate.reliability(y, scored["p_raw"]), "calibrated": evaluate.reliability(y, scored["p_cal"])}
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = (1e-4, 1)
    ax.plot(lim, lim, color=viz.INK2, lw=1, ls="--", label="perfect calibration")
    for j, (name, t) in enumerate(tabs.items()):
        t = t[(t["mean_pred"] > 0) & (t["observed"] > 0)]
        ax.plot(t["mean_pred"], t["observed"], marker="o", color=viz.SERIES[j], lw=2,
                label=f"{name} (ECE {evaluate.ece(y, scored['p_raw' if name == 'raw' else 'p_cal']):.4f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed winner rate (bin)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Winner classifier reliability, {scored['cutoff'].nunique()} out-of-time cutoffs (pooled)")
    fig.savefig(path)
    plt.close(fig)
    return pd.concat({k: v for k, v in tabs.items()}, names=["probability"]).reset_index(level=0)




def tag(share: float) -> str:
    """File-name suffix for a label share ('' for the default top 1%, e.g. '_top0.1pct' otherwise)."""
    return "" if share == WINNER_SHARE else f"_top{share * 100:g}pct"


def paths(share: float) -> dict[str, Path]:
    t = tag(share)
    return {"oof": OUT_DIR / f"oof_predictions{t}.parquet", "per_cutoff": OUT_DIR / f"per_cutoff{t}.csv",
            "report": OUT_DIR / f"eval_classifier{t}.md", "plot": OUT_DIR / f"reliability{t}.png",
            "scores": OUT_DIR / f"scores_clf{t}.parquet",
            "model": config.MODELS_DIR / f"lgbm_clf{t}_final_{config.FINAL_CUTOFF:%Y%m%d}.txt",
            "calibrator": config.MODELS_DIR / f"isotonic_clf{t}_{config.FINAL_CUTOFF:%Y%m%d}.json",
            "meta": config.MODELS_DIR / f"clf{t}_meta.json"}


def score_final(all_df: pd.DataFrame, rounds: int, cal: evaluate.Isotonic) -> tuple[pd.DataFrame, lgb.Booster]:
    """Train on every cutoff with a fully observed target (2019-09-25 .. 2020-08-26) and score all styles
    active at FINAL_CUTOFF (forecast window 2020-09-23 .. 2020-10-20). Adds the regressor's unit forecast."""
    from forecasting import data, select

    c = pd.Timestamp(config.FINAL_CUTOFF)
    booster = fit(history_for(all_df, c), rounds)
    snap = features.make_snapshot(config.FINAL_CUTOFF)
    out = snap[["product_code", "units_w1", "units_w2", "units_w4"]].copy()
    out["p_raw"] = predict_proba(booster, snap)
    out["p_cal"] = cal.predict(out["p_raw"])
    reg = select.get_predictions(config.FINAL_CUTOFF)[["product_code", "pred_units"]]
    out = out.merge(reg.rename(columns={"pred_units": "reg_pred_units"}), on="product_code", how="left")
    attrs = data.style_attributes(as_of=config.FINAL_CUTOFF)[["product_code", "prod_name", "product_type_name",
                                                              "garment_group_name", "index_group_name"]]
    out = out.merge(attrs, on="product_code", how="left")
    for col in ("product_type_name", "garment_group_name", "index_group_name"):
        out[col] = out[col].astype(str)
    out.insert(1, "cutoff", c)
    out["rank_raw"] = out["p_raw"].rank(method="first", ascending=False).astype(int)
    out["reg_rank"] = out["reg_pred_units"].rank(method="first", ascending=False).astype(int)
    return out.sort_values("rank_raw", ignore_index=True), booster


def write_report(share: float, pc: pd.DataFrame, rel: pd.DataFrame, rounds: int, n_oof_cal: int,
                 scores: pd.DataFrame) -> str:
    bt = pc[pc["cutoff"] < pd.Timestamp(config.VALID_CUTOFF)]
    va = pc[pc["cutoff"] == pd.Timestamp(config.VALID_CUTOFF)]
    clf = pc[pc["method"] == CLF]
    cuts = sorted(bt["cutoff"].unique())
    br = clf.drop_duplicates("cutoff")
    label = f"top {share * 100:g}%"

    def mean(m, k, d=bt):
        return d.loc[d["method"] == m, k].mean()

    def wins(m, k, d=bt):
        p = d.pivot(index="cutoff", columns="method", values=k)
        return int(((p[CLF] - p[m]) > 1e-12).sum())

    verdict = []
    for k in RANK_METRICS:
        c, b = mean(CLF, k), mean(L1, k)
        word = "higher" if c > b + 1e-12 else ("equal" if abs(c - b) <= 1e-12 else "lower")
        verdict.append(f"{k}: classifier {c:.3f} vs last week × 4 {b:.3f} ({word}; classifier ahead at "
                       f"{wins(L1, k)}/{len(cuts)} cutoffs)")
    cal_rows = [("raw", "brier_raw", "ece_raw"), ("isotonic-calibrated", "brier_cal", "ece_cal")]
    top = scores.nsmallest(20, "reg_rank")
    md = [
        f"# Winner classifier ({label} label): evaluation",
        "",
        f"Success = a style's units over the next 4 weeks rank in the **{label} of styles active at that cutoff** "
        "(threshold computed per cutoff; ties at the threshold count as winners). Model: LightGBM binary on the "
        f"regressor's 23 features, {rounds} boosting rounds (early stopping on the last 4 training cutoffs, "
        "metric average precision).",
        "",
        "## Winner base rate",
        "",
        f"- Mean base rate over the {len(br)} scored cutoffs: **{br['base_rate'].mean():.4%}** "
        f"({br['n_winners'].mean():.0f} winners out of {br['n_styles'].mean():,.0f} active styles on average; "
        f"range {br['n_winners'].min()}–{br['n_winners'].max()} winners).",
        f"- Units needed to be a winner (next 4 weeks): {br['threshold_units'].min():,.0f}–"
        f"{br['threshold_units'].max():,.0f} depending on the cutoff (median {br['threshold_units'].median():,.0f}).",
        "",
        f"## Rolling backtest: {len(cuts)} weekly cutoffs ({cuts[0]:%Y-%m-%d} → {cuts[-1]:%Y-%m-%d}), mean ± std",
        "",
        "Same folds as the regressor's backtest: at cutoff c every model (classifier and regressor) is retrained "
        "only on cutoffs whose 4-week target ended by c. Every method is ranked by its own score; the classifier "
        "by raw probability.",
        "",
        *_summary_table(bt),
        "",
        f"## Validation cutoff {config.VALID_CUTOFF} (trained on 2019-09-25 → 2020-07-29)",
        "",
        *_valid_table(va),
        "",
        "## Does the classifier beat last week × 4? (backtest means)",
        "",
        *[f"- {v}" for v in verdict],
        "",
        "## Calibration (out-of-time)",
        "",
        "The isotonic map used at each cutoff is fitted only on out-of-fold predictions whose labels were known "
        f"at that cutoff (OOF cutoff ≤ cutoff − 4 weeks). {N_WARMUP} extra OOF cutoffs before the backtest window "
        "are used only as calibration data. Brier and ECE per cutoff, mean over the cutoffs:",
        "",
        "| probabilities | Brier (backtest) | ECE (backtest) | Brier (validation) | ECE (validation) |",
        "|---|---|---|---|---|",
        *[f"| {name} | {mean(CLF, b):.5f} | {mean(CLF, e):.5f} | {mean(CLF, b, va):.5f} | {mean(CLF, e, va):.5f} |"
          for name, b, e in cal_rows],
        "",
        f"Calibrated beats raw on Brier at {int((clf['brier_cal'] < clf['brier_raw']).sum())}/{len(clf)} cutoffs "
        f"and on ECE at {int((clf['ece_cal'] < clf['ece_raw']).sum())}/{len(clf)}.",
        "",
        f"Pooled reliability over all scored cutoffs ({paths(share)['plot'].name}):",
        "",
        "| probabilities | bin | n | mean predicted | observed |",
        "|---|---|---|---|---|",
        *[f"| {r.probability} | {r.bin} | {r.n:,} | {r.mean_pred:.4f} | {r.observed:.4f} |" for r in rel.itertuples()],
        "",
        "## Final scoring: cutoff 2020-09-23 (forecast window 2020-09-23 → 2020-10-20)",
        "",
        f"Classifier retrained on all cutoffs 2019-09-25 → 2020-08-26; calibrator fitted on {n_oof_cal:,} OOF rows "
        "(cutoffs up to 2020-08-26, targets ending by 2020-09-22). Among the regressor's top-20 styles, "
        f"{int((top['p_cal'] >= 0.999).sum())} have a calibrated probability ≥ 0.999 and there are "
        f"{top['p_cal'].round(4).nunique()} distinct calibrated values (range {top['p_cal'].min():.3f}–"
        f"{top['p_cal'].max():.3f}). The final ranking and prediction_score are set in final_scores.md.",
        "",
        "Metric definitions: pr_auc = average precision (tie-aware); precision@k = share of winners among the k "
        "highest scores; recall@50 = share of all winners found in the top 50; Brier = mean squared error of the "
        "probability; ECE = n-weighted mean |predicted − observed| over the bins in evaluate.CAL_BINS.",
    ]
    text = "\n".join(md) + "\n"
    paths(share)["report"].write_text(text, encoding="utf-8")
    return text


def report(share: float) -> str:
    """Rebuild per_cutoff.csv, the reliability plot and the markdown report from saved OOF predictions."""
    p = paths(share)
    oof = pd.read_parquet(p["oof"])
    scored = oof[oof["p_cal"].notna()].reset_index(drop=True)
    pc = per_cutoff_metrics(scored)
    pc.to_csv(p["per_cutoff"], index=False)
    rel = reliability_plot(scored, p["plot"])
    meta = json.loads(p["meta"].read_text())
    return write_report(share, pc, rel, meta["rounds"], meta["calibrator_oof_rows"], pd.read_parquet(p["scores"]))


def run(share: float) -> None:
    """Train, backtest, calibrate and score the final cutoff for one label share; then write its report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = paths(share)
    train = pd.read_parquet(features.TRAIN_PATH)
    valid = pd.read_parquet(features.VALID_PATH)
    all_df = pd.concat([train, valid], ignore_index=True)
    all_df["y"] = winner_labels(all_df, share)
    train = all_df[all_df["cutoff"] <= pd.Timestamp(features.TRAIN_LAST_CUTOFF)]

    t = time.time()
    rounds = best_rounds(train)
    print(f"early stopping: {rounds} rounds ({time.time() - t:.0f}s)", flush=True)
    reg_rounds = json.loads(model.META_PATH.read_text())["best_iterations"]["regressor"]

    train_cuts = sorted(train["cutoff"].unique())
    bt_cuts = train_cuts[-N_BACKTEST:]
    warm_cuts = train_cuts[-N_BACKTEST - N_WARMUP:-N_BACKTEST]
    eval_cuts = [*bt_cuts, pd.Timestamp(config.VALID_CUTOFF)]
    warm = out_of_fold(all_df, warm_cuts, rounds, reg_rounds=None)
    scored = out_of_fold(all_df, eval_cuts, rounds, reg_rounds=reg_rounds)
    oof = pd.concat([warm, scored], ignore_index=True)

    cal_parts = []
    for c in eval_cuts:
        g = scored[scored["cutoff"] == c]
        cal_parts.append(pd.Series(fit_calibrator(oof, c).predict(g["p_raw"]), index=g.index))
    scored["p_cal"] = pd.concat(cal_parts)
    oof = oof.merge(scored[["product_code", "cutoff", "p_cal"]], on=["product_code", "cutoff"], how="left")
    oof.to_parquet(p["oof"], index=False)

    final_cal = fit_calibrator(oof, pd.Timestamp(config.FINAL_CUTOFF))
    n_oof_cal = len(history_for(oof, pd.Timestamp(config.FINAL_CUTOFF)))
    scores, booster = score_final(all_df, rounds, final_cal)
    scores.to_parquet(p["scores"], index=False)
    booster.save_model(str(p["model"]))
    p["calibrator"].write_text(json.dumps(final_cal.to_dict()))
    p["meta"].write_text(json.dumps({
        "label": f"top {share * 100:g}% of active styles by next-{config.HORIZON_WEEKS}-week units, per cutoff",
        "share": share, "rounds": rounds, "params": PARAMS, "features": features.FEATURES,
        "train_cutoffs": [str(features.TRAIN_FIRST_CUTOFF), str(config.VALID_CUTOFF)],
        "calibrator_oof_rows": n_oof_cal}, indent=2, default=str))
    print(report(share))


def select_top_k(scores: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Published rule (select.select_top_k): highest regressor forecast units, sold in the last 2 weeks, at most
    one style per garment group."""
    cand = scores[scores["units_w2"] > 0].sort_values("forecast_units", ascending=False)
    out = cand.drop_duplicates("garment_group_name", keep="first").head(k).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def combine(strict: float = STRICT_SHARE, top_n: int = 20) -> pd.DataFrame:
    """Pick prediction_score and write scores_final.parquet + final_scores.md.

    Ranking stays the regressor's forecast units (the published top-3). prediction_score = calibrated
    P(top ``strict``) when it is usable, else calibrated P(top 1%) (ties broken by forecast units). Usable means:
    among the regressor's top ``top_n`` it is not saturated (at most 3 values >= 0.999 and at least 10 distinct
    values), its isotonic calibration does not raise the backtest ECE, and every cutoff has >= 10 winners.
    """
    s1 = pd.read_parquet(paths(WINNER_SHARE)["scores"])
    s2 = pd.read_parquet(paths(strict)["scores"])[["product_code", "p_raw", "p_cal"]]
    pc2 = pd.read_csv(paths(strict)["per_cutoff"], parse_dates=["cutoff"])
    df = (s1.rename(columns={"p_raw": "p_raw_top1pct", "p_cal": "confidence_top1pct", "reg_pred_units": "forecast_units"})
          .merge(s2.rename(columns={"p_raw": "p_raw_strict", "p_cal": "p_strict"}), on="product_code", how="left"))
    top = df.nsmallest(top_n, "reg_rank")
    clf2 = pc2[pc2["method"] == CLF]
    bt2 = clf2[clf2["cutoff"] < pd.Timestamp(config.VALID_CUTOFF)]
    n_sat = int((top["p_strict"] >= 0.999).sum())
    n_distinct = int(top["p_strict"].round(4).nunique())
    ece_raw, ece_cal = bt2["ece_raw"].mean(), bt2["ece_cal"].mean()
    min_pos = int(clf2["n_winners"].min())
    checks = {"not saturated": n_sat <= 3 and n_distinct >= 10, "calibration does not raise ECE": ece_cal <= ece_raw,
              ">= 10 winners per cutoff": min_pos >= 10}
    use_strict = all(checks.values())
    strict_label = f"top {strict * 100:g}%"
    df["prediction_score"] = df["p_strict"] if use_strict else df["confidence_top1pct"]
    df["score_source"] = f"calibrated P({strict_label})" if use_strict else "calibrated P(top 1%)"
    df = df.sort_values(["forecast_units"], ascending=False, ignore_index=True)
    top3 = select_top_k(df)
    df["rank"] = df["product_code"].map(dict(zip(top3["product_code"], top3["rank"])))
    df.to_parquet(OUT_DIR / "scores_final.parquet", index=False)

    t10 = df.head(10)
    md = ["# Final scores at cutoff 2020-09-23 (forecast window 2020-09-23 → 2020-10-20)", "",
          "Ranking = the regressor's forecast units (published rule: one style per garment group, sold in the last "
          "2 weeks). The classifiers only supply probabilities.", "",
          f"## Is calibrated P({strict_label}) usable as prediction_score?", "",
          f"- Among the regressor's top {top_n}: {n_sat} styles at ≥ 0.999, {n_distinct} distinct values "
          f"(range {top['p_strict'].min():.3f}–{top['p_strict'].max():.3f}) → "
          f"{'spread out' if checks['not saturated'] else 'saturated'}.",
          f"- Backtest ECE raw {ece_raw:.5f} vs calibrated {ece_cal:.5f}.",
          f"- Fewest winners at any scored cutoff: {min_pos}.",
          f"- Decision: prediction_score = **{df['score_source'].iat[0]}**"
          + ("" if use_strict else " (ties broken by forecast units)") + "; confidence_top1pct = calibrated P(top 1%).",
          "", "## Top 10 by regressor forecast units", "",
          f"| reg rank | style | name | garment group | forecast units | P({strict_label}) calibrated | "
          "P(top 1%) calibrated | top-3 |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.reg_rank} | {r.product_code} | {r.prod_name} | {r.garment_group_name} | {r.forecast_units:,.0f} "
           f"| {r.p_strict:.3f} | {r.confidence_top1pct:.3f} | {'' if pd.isna(r.rank) else f'#{int(r.rank)}'} |"
           for r in t10.itertuples()]
    md += ["", "Top-3: " + ", ".join(f"#{r.rank} {r.product_code} {r.prod_name} (prediction_score "
                                     f"{r.prediction_score:.3f}, {r.forecast_units:,.0f} units)" for r in top3.itertuples())]
    if top3["product_code"].tolist() != PUBLISHED_TOP3:
        md.append(f"\n**Warning:** top-3 differs from the published list {PUBLISHED_TOP3}.")
    (OUT_DIR / "final_scores.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--share", type=float, default=WINNER_SHARE, help="winner label share (0.01 = top 1%%)")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from saved OOF predictions")
    ap.add_argument("--combine", action="store_true", help="choose prediction_score and write scores_final")
    a = ap.parse_args()
    if a.combine:
        combine()
    elif a.report_only:
        print(report(a.share))
    else:
        run(a.share)
