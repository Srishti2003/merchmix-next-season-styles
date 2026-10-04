"""LightGBM regressor / ranker training, persistence and SHAP explanations.

Two candidates, both trained on the weekly snapshots from data_science.features:
- regressor: objective='regression' on the *uplift over the naive run-rate*:
             y_log - log1p(4 x last-week units), weighted by 1 + log1p(units_w4) so the high-volume
             styles we rank at the top count more. Prediction adds the run-rate back -> unit forecast.
             (Plain y_log lost to "last week x 4" at the top of the ranking; see eval_table.md.)
- ranker:    lambdarank, label = 0..30 quantile bucket of y_units within the cutoff. LightGBM caps a
             query at 10k rows, so each cutoff (~20k styles) is split into N_SHARDS deterministic
             shards (product_code % N_SHARDS); query group = (cutoff, shard).
Early stopping uses the last 4 training cutoffs as an inner validation set; each model is then
refit on all training cutoffs with its best iteration count. The candidate with the better
inner-backtest NDCG@50 becomes the primary model (recorded in lgbm_meta.json). The regressor is
always saved too (lgbm_reg.txt) because it is the only one that forecasts units.
"""
from __future__ import annotations

import json
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from data_science import data, evaluate, features

RANKER_PATH = config.MODELS_DIR / "lgbm_ranker.txt"  # written only if the ranker ever wins
REG_PATH = config.MODELS_DIR / "lgbm_reg.txt"
META_PATH = config.MODELS_DIR / "lgbm_meta.json"
N_INNER = 4
N_SHARDS = 3
N_BACKTEST = 10

_BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20,
             seed=config.SEED, deterministic=True, force_row_wise=True,
             verbose=-1, num_threads=0)
PARAMS: dict[str, dict] = {
    "regressor": {**_BASE, "objective": "regression", "metric": "l2"},
    "ranker": {**_BASE, "objective": "lambdarank", "metric": "ndcg", "eval_at": [50],
               "lambdarank_truncation_level": 100},
}

FEATURE_LABELS: dict[str, str] = {
    "units_w1": "units sold last week", "units_w2": "units sold in the last 2 weeks",
    "units_w4": "units sold in the last 4 weeks", "units_w8": "units sold in the last 8 weeks",
    "units_w12": "units sold in the last 12 weeks",
    "trend_1_4": "last week vs 4-week average (momentum)", "trend_4_12": "4-week vs 12-week average (trend)",
    "buyers_w4": "distinct buyers per week, last 4 weeks (summed)", "repeat_rate_w4": "units per buyer",
    "online_share_w4": "online share of units", "n_active_articles_w4": "colour/size variants selling",
    "avg_price_w4": "average price last 4 weeks", "discount": "discount vs highest price seen",
    "weeks_since_launch": "weeks since first sale", "is_new": "launched < 8 weeks ago",
    "ly_units_h": "units in the same 4 weeks last year", "week_of_year": "week of year",
    "product_type_name": "product type", "product_group_name": "product group",
    "garment_group_name": "garment group", "index_group_name": "index group",
    "colour_group_name": "main colour", "graphical_appearance_name": "graphical appearance",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in model column order, categoricals as pandas 'category'."""
    X = df[features.FEATURES].copy()
    for c in features.CAT_FEATURES:
        X[c] = X[c].astype(str).astype("category")
    return X


def rank_labels(df: pd.DataFrame, n_buckets: int = 30) -> np.ndarray:
    """0..n_buckets quantile bucket of y_units within each cutoff (ties share a bucket)."""
    pct = df.groupby("cutoff")["y_units"].rank(pct=True, method="max")
    lab = np.floor(pct * n_buckets).astype(int)
    return np.where(df["y_units"].to_numpy() == 0, 0, lab)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows so ranker query groups (cutoff, shard) are contiguous."""
    shard = df["product_code"].astype(int) % N_SHARDS
    out = df.assign(_shard=shard).sort_values(["cutoff", "_shard", "product_code"], kind="stable")
    return out.drop(columns="_shard").reset_index(drop=True)


def run_rate_log(df: pd.DataFrame) -> np.ndarray:
    """Naive run-rate forecast in log space: log1p(4 x last-week units)."""
    return np.log1p(4 * df["units_w1"].to_numpy(dtype=float))


def _dataset(df: pd.DataFrame, kind: str, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """LightGBM Dataset for ``kind`` (``df`` must be ordered by ``_sort`` for the ranker)."""
    if kind == "regressor":
        return lgb.Dataset(prepare(df), df["y_log"].to_numpy() - run_rate_log(df),
                           weight=1 + np.log1p(df["units_w4"].to_numpy(dtype=float)),
                           reference=reference, free_raw_data=False)
    groups = df.groupby([df["cutoff"], df["product_code"].astype(int) % N_SHARDS], sort=True).size().to_numpy()
    return lgb.Dataset(prepare(df), rank_labels(df), group=groups, reference=reference, free_raw_data=False)


def fit(train: pd.DataFrame, kind: str) -> tuple[lgb.Booster, lgb.Booster, int]:
    """Train ``kind`` with inner early stopping, then refit on all of ``train``.

    Returns (inner_model trained without the last N_INNER cutoffs, full_model, best_iteration).
    """
    train = _sort(train)
    cut = sorted(train["cutoff"].unique())[-N_INNER]
    tr, va = train[train["cutoff"] < cut], train[train["cutoff"] >= cut]
    dtr = _dataset(tr, kind)
    dva = _dataset(va, kind, reference=dtr)
    inner = lgb.train(PARAMS[kind], dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    best = inner.best_iteration
    full = lgb.train(PARAMS[kind], _dataset(train, kind), num_boost_round=best)
    return inner, full, best


def predict(booster: lgb.Booster, kind: str, snap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (score, unit forecast or None) for a snapshot."""
    s = booster.predict(prepare(snap))
    if kind != "regressor":
        return s, None
    s = s + run_rate_log(snap)  # back to log1p(units)
    return s, np.expm1(s).clip(min=0)


def inner_backtest(inner: lgb.Booster, kind: str, train: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics of the inner model and baselines over the last N_INNER training cutoffs."""
    tables = []
    for c in sorted(train["cutoff"].unique())[-N_INNER:]:
        snap = train[train["cutoff"] == c].reset_index(drop=True)
        tables.append(evaluate.compare(snap, {f"LightGBM {kind}": predict(inner, kind, snap)}))
    return sum(tables) / len(tables)


def rolling_backtest(train: pd.DataFrame, rounds: int, n: int = 10) -> pd.DataFrame:
    """Rolling-origin backtest of the regressor over the last ``n`` training cutoffs.

    For each cutoff c the model is retrained only on cutoffs whose target window ends by c
    (cutoff <= c - horizon), with a fixed ``rounds`` (from early stopping), and scored at c.
    Returns long-form metrics: one row per (cutoff, method).
    """
    rows = []
    for c in sorted(train["cutoff"].unique())[-n:]:
        tr = train[train["cutoff"] <= c - pd.Timedelta(weeks=config.HORIZON_WEEKS)]
        m = lgb.train(PARAMS["regressor"], _dataset(tr, "regressor"), num_boost_round=rounds)
        snap = train[train["cutoff"] == c].reset_index(drop=True)
        t = evaluate.compare(snap, {"LightGBM regressor": predict(m, "regressor", snap)})
        rows.append(t.assign(cutoff=c).rename_axis("method").reset_index())
    return pd.concat(rows, ignore_index=True)


def backtest_summary(bt: pd.DataFrame, methods: list[str], metrics: list[str]) -> str:
    """Markdown: mean ± std per method/metric, plus model win counts vs each baseline."""
    n = bt["cutoff"].nunique()
    lines = ["| method | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mth in methods:
        g = bt[bt["method"] == mth]
        lines.append(f"| {mth} | " + " | ".join(f"{g[m].mean():.3f} ± {g[m].std():.3f}" for m in metrics) + " |")
    piv = {m: bt.pivot(index="cutoff", columns="method", values=m) for m in metrics}
    lines.append("")
    for base in methods[1:]:
        parts = []
        for m in metrics:
            d = piv[m]["LightGBM regressor"] - piv[m][base]
            parts.append(f"{m}: model wins **{(d > 1e-12).sum()}/{n}** (ties {(d.abs() <= 1e-12).sum()})")
        lines.append(f"- vs {base} — " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def save(models: dict[str, lgb.Booster], primary: str, best_iters: dict[str, int], extra: dict) -> None:
    """Persist the regressor (lgbm_reg.txt; also the ranker if it is primary) and metadata."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models["regressor"].save_model(str(REG_PATH))
    if primary == "ranker":
        models["ranker"].save_model(str(RANKER_PATH))
    META_PATH.write_text(json.dumps({"primary": primary, "best_iterations": best_iters,
                                     "features": features.FEATURES, **extra}, indent=2, default=str))


def load(kind: str | None = None) -> tuple[lgb.Booster, str]:
    """Load the primary model (``kind=None``) or the regressor (``kind='regressor'``)."""
    meta = json.loads(META_PATH.read_text())
    if kind == "regressor":
        return lgb.Booster(model_file=str(REG_PATH)), "regressor"
    path = REG_PATH if meta["primary"] == "regressor" else RANKER_PATH
    return lgb.Booster(model_file=str(path)), meta["primary"]


def shap_values(booster: lgb.Booster, snap: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP contributions (n_rows x n_features) and the base value, via LightGBM's TreeSHAP."""
    contrib = booster.predict(prepare(snap), pred_contrib=True)
    return contrib[:, :-1], float(contrib[0, -1])


def _fmt(feat: str, v: object) -> str:
    """Human-readable feature value."""
    if feat in ("online_share_w4", "discount"):
        return f"{float(v):.0%}"
    if feat == "avg_price_w4":
        return f"{float(v):.4f} (scaled)"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):,.2f}".rstrip("0").rstrip(".") if not float(v).is_integer() else f"{int(v):,}"
    return str(v)


def explain(product_code: str, cutoff: date = config.FINAL_CUTOFF, n: int = 5,
            snap: pd.DataFrame | None = None, booster: lgb.Booster | None = None) -> list[str]:
    """Top-``n`` SHAP drivers of the regressor's forecast for one style, in plain English.

    The regressor predicts the log-uplift over the naive run-rate (last week x 4), so each driver
    is a multiplier the model applies on top of that run-rate.
    """
    booster = booster or load("regressor")[0]
    snap = features.make_snapshot(cutoff) if snap is None else snap
    row = snap[snap["product_code"] == str(product_code).zfill(7)].reset_index(drop=True)
    if row.empty:
        raise ValueError(f"Style {product_code} has no sales in the 12 weeks before {cutoff}.")
    sv, base = shap_values(booster, row)
    order = np.argsort(-np.abs(sv[0]))[:n]
    out = []
    for i in order:
        f = features.FEATURES[i]
        mult = np.exp(sv[0, i])
        direction = "raises" if sv[0, i] > 0 else "lowers"
        out.append(f"{FEATURE_LABELS[f]} = {_fmt(f, row.at[0, f])} → {direction} the forecast "
                   f"×{mult:.2f} vs the last-week run-rate")
    return out


def shap_summary_plot(booster: lgb.Booster, snap: pd.DataFrame, path, n: int = 5000) -> None:
    """Save a SHAP beeswarm summary for a random sample of ``snap``."""
    import matplotlib.pyplot as plt
    import shap

    sample = snap.sample(min(n, len(snap)), random_state=config.SEED).reset_index(drop=True)
    sv, _ = shap_values(booster, sample)
    X = prepare(sample)
    for c in features.CAT_FEATURES:  # beeswarm needs numbers; colour categoricals by code
        X[c] = X[c].cat.codes
    shap.summary_plot(sv, X, feature_names=[FEATURE_LABELS[f] for f in features.FEATURES],
                      show=False, max_display=15, plot_size=(9, 6))
    plt.title("What moves the forecast away from 'last week × 4' (SHAP, log-uplift)", loc="left", fontweight="bold")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")


def shap_waterfall(product_code: str, cutoff: date, path) -> None:
    """Save a SHAP waterfall plot for one style's regressor forecast."""
    import matplotlib.pyplot as plt
    import shap

    booster, _ = load("regressor")
    snap = features.make_snapshot(cutoff)
    row = snap[snap["product_code"] == str(product_code).zfill(7)].reset_index(drop=True)
    sv, base = shap_values(booster, row)
    ex = shap.Explanation(values=sv[0], base_values=base, data=row[features.FEATURES].iloc[0].to_numpy(),
                          feature_names=[FEATURE_LABELS[f] for f in features.FEATURES])
    shap.plots.waterfall(ex, max_display=8, show=False)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")


def main() -> None:
    """Train both candidates, evaluate vs baselines, pick the primary model, write reports."""
    import time

    import matplotlib.pyplot as plt

    from data_science import viz

    train = pd.read_parquet(features.TRAIN_PATH)
    valid = pd.read_parquet(features.VALID_PATH)
    models, iters, backtests, scores = {}, {}, [], {}
    for kind in ("regressor", "ranker"):
        t = time.time()
        inner, full, best = fit(train, kind)
        models[kind], iters[kind] = full, best
        backtests.append(inner_backtest(inner, kind, train))
        scores[f"LightGBM {kind}"] = predict(full, kind, valid)
        print(f"{kind}: best_iter={best}, {time.time() - t:.0f}s")

    bt = pd.concat(backtests)
    bt = bt[~bt.index.duplicated()]  # baseline rows appear once per candidate
    primary = max(("regressor", "ranker"), key=lambda k: bt.loc[f"LightGBM {k}", "ndcg@50"])
    table = evaluate.compare(valid, scores)
    save(models, primary, iters, {"valid_cutoff": config.VALID_CUTOFF})

    t = time.time()
    roll = rolling_backtest(train, iters["regressor"], n=N_BACKTEST)
    print(f"rolling backtest ({N_BACKTEST} cutoffs): {time.time() - t:.0f}s")
    bt_methods = ["LightGBM regressor", evaluate.BASELINES["last_1w_x4"], evaluate.BASELINES["last_4w"]]
    bt_metrics = ["ndcg@50", "precision@12"]
    cuts = sorted(roll["cutoff"].unique())

    md = ["# Evaluation: LightGBM vs naive baselines", "",
          f"Primary model: **LightGBM {primary}** — chosen on the inner early-stopping cutoffs "
          f"(NDCG@50 regressor {bt.loc['LightGBM regressor', 'ndcg@50']:.3f} vs ranker "
          f"{bt.loc['LightGBM ranker', 'ndcg@50']:.3f}), not on the validation week, to avoid selection bias.", "",
          evaluate.to_markdown(table, f"Validation cutoff {config.VALID_CUTOFF} "
                                      f"(target = units {config.VALID_CUTOFF} → 2020-09-22, "
                                      f"{len(valid):,} styles)"),
          f"### Rolling backtest: {N_BACKTEST} weekly cutoffs "
          f"({pd.Timestamp(cuts[0]):%Y-%m-%d} → {pd.Timestamp(cuts[-1]):%Y-%m-%d}), mean ± std", "",
          "At each cutoff the regressor is retrained only on cutoffs whose 4-week target ended before it "
          f"({iters['regressor']} boosting rounds, fixed).", "",
          backtest_summary(roll, bt_methods, bt_metrics),
          "The regressor predicts log-uplift over the naive 'last week × 4' run-rate, weighted toward "
          "high-volume styles (a plain log-units regressor lost to that baseline at the top of the list). "
          "Where the model disagrees with that baseline and who was right: see movers.md.", "",
          f"precision@k = share of predicted top-k in actual top-k; {evaluate.HIT_RATE} = share of the actual "
          "top-12 that appear in the predicted top-50; WAPE is n/a for the ranker (no unit forecast)."]
    (config.FIGURES_DIR / "eval_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.round(3).to_string())
    print(backtest_summary(roll, bt_methods, bt_metrics))

    # --- movers: model vs "last week × 4" at the validation cutoff ---
    promoted, demoted, lines = evaluate.movers(valid, scores["LightGBM regressor"][0])
    names = data.style_attributes()[["product_code", "prod_name", "product_type_name"]]
    mv = ["# Movers: where the model disagrees with “last week × 4”", "",
          f"Validation cutoff {config.VALID_CUTOFF}; pool = styles in the top-100 of either ranking. "
          "“Right?” = the actual rank is closer (in log-rank) to the model's rank than to the baseline's.", "",
          *[f"- {s}" for s in lines], ""]
    for title, g in (("Model ranks most ABOVE the baseline", promoted), ("Model ranks most BELOW the baseline", demoted)):
        g = g.merge(names, on="product_code", how="left")
        mv += [f"### {title}", "", "| style | name | type | model rank | baseline rank | actual rank | actual units | right? |",
               "|---|---|---|---|---|---|---|---|"]
        mv += [f"| {r.product_code} | {r.prod_name} | {r.product_type_name} | {r.model_rank} | {r.baseline_rank} "
               f"| {r.actual_rank} | {r.y_units:,.0f} | {'✓' if r.model_right else '✗'} |" for r in g.itertuples()]
        mv.append("")
    (config.FIGURES_DIR / "movers.md").write_text("\n".join(mv), encoding="utf-8")
    print("\n".join(lines))

    # --- chart: validation week (left) + rolling backtest mean ± std (right) ---
    viz.apply_style()
    colors = {name: viz.SERIES[j] for j, name in enumerate(table.index)}
    metrics = ["precision@3", "precision@12", evaluate.HIT_RATE, "ndcg@50", "spearman_top500"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.8 / len(table)
    x = np.arange(len(metrics))
    for j, (name, r) in enumerate(table.iterrows()):
        ax.bar(x + (j - (len(table) - 1) / 2) * w, r[metrics], width=w - 0.02, label=name,
               color=colors[name], edgecolor=viz.SURFACE, linewidth=1)
    ax.set_xticks(x, [m.replace(" in ", "\nin ") for m in metrics])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.05)
    fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=9)
    ax.set_title(f"Validation week (cutoff {config.VALID_CUTOFF})", pad=10)
    w2 = 0.8 / len(bt_methods)
    x2 = np.arange(len(bt_metrics))
    for j, mth in enumerate(bt_methods):
        g = roll[roll["method"] == mth]
        ax2.bar(x2 + (j - (len(bt_methods) - 1) / 2) * w2, g[bt_metrics].mean(), yerr=g[bt_metrics].std(),
                width=w2 - 0.02, color=colors[mth], edgecolor=viz.SURFACE, linewidth=1,
                error_kw={"ecolor": viz.INK2, "elinewidth": 1, "capsize": 3})
    ax2.set_xticks(x2, bt_metrics)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Rolling backtest, {N_BACKTEST} cutoffs (mean ± std)", pad=10)
    fig.suptitle("LightGBM vs naive baselines (higher = better)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(config.FIGURES_DIR / "eval_vs_baselines.png")
    plt.close(fig)

    shap_summary_plot(models["regressor"], valid, config.FIGURES_DIR / "shap_summary.png")
    top = valid.assign(p=scores["LightGBM regressor"][0]).nlargest(1, "p")["product_code"].iat[0]
    print(f"\nexplain({top}, {config.VALID_CUTOFF}):")
    for s in explain(top, config.VALID_CUTOFF, snap=valid):
        print("  -", s)


if __name__ == "__main__":
    main()
