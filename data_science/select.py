"""Final prediction, diversified top-k selection and evidence export.

Pipeline for a prediction cutoff ``c`` (a Wednesday; FINAL_CUTOFF = 2020-09-23 = day after the data):
1. Retrain the regressor on every weekly cutoff whose 4-week target is fully observed before ``c``
   (for FINAL_CUTOFF: 2019-09-25 .. 2020-08-26, target ending 2020-09-22), using the number of
   boosting rounds chosen by early stopping in data_science.model.
2. Predict next-4-week units for every style active in the 12 weeks before ``c``.
3. select_top_k: highest predicted units, at most one style per garment group, and the style must
   have sold in the last 2 weeks. That last rule is our ONLY availability signal: the dataset has no
   stock data, so "still selling" is a proxy for "still in stock / still ranged".
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from data_science import data, features, model

KAGGLE_COMP = "h-and-m-personalized-fashion-recommendations"
ATTR_COLS = ["prod_name", "product_type_name", "product_group_name", "garment_group_name",
             "index_group_name", "section_name", "colour_group_name", "graphical_appearance_name",
             "n_colours", "detail_desc"]


def train_final(cutoff: date) -> lgb.Booster:
    """Train the regressor on all cutoffs whose target window ends by ``cutoff``."""
    last = cutoff - timedelta(weeks=config.HORIZON_WEEKS)
    cutoffs = features.default_cutoffs(features.TRAIN_FIRST_CUTOFF, last)
    train = features.build_training_set(cutoffs, path=None)
    rounds = json.loads(model.META_PATH.read_text())["best_iterations"]["regressor"]
    return lgb.train(model.PARAMS["regressor"], model._dataset(train, "regressor"), num_boost_round=rounds)


def normalize_cutoff(d: date | str) -> date:
    """Turn a user-facing date into a valid exclusive cutoff (a Wednesday).

    A Wednesday is returned as is. Any other date means "use data up to and including this day",
    so it maps to the day after the last complete Wed->Tue week ending on or before it
    (e.g. Tue 2020-09-22 -> Wed 2020-09-23; Mon 2020-09-21 -> Wed 2020-09-16). Never leaks later data.
    """
    d = date.fromisoformat(d) if isinstance(d, str) else d
    if d.weekday() != 2:
        d = d - timedelta(days=(d.weekday() - 1) % 7) + timedelta(days=1)  # last Tuesday <= d, +1
    first_ok = features.TRAIN_FIRST_CUTOFF + timedelta(weeks=12)
    if not first_ok <= d <= config.FINAL_CUTOFF:
        raise ValueError(f"Cutoff {d} out of range: it must lie between {first_ok} and {config.FINAL_CUTOFF} "
                         "(data ends 2020-09-22; the model needs ~6 months of training snapshots before the cutoff).")
    return d


def get_booster(cutoff: date, refresh: bool = False) -> lgb.Booster:
    """Final model for ``cutoff``, cached at models/lgbm_final_<YYYYMMDD>.txt."""
    path = config.MODELS_DIR / f"lgbm_final_{cutoff:%Y%m%d}.txt"
    if path.exists() and not refresh:
        return lgb.Booster(model_file=str(path))
    booster = train_final(cutoff)
    booster.save_model(str(path))
    return booster


def get_predictions(cutoff: date, refresh: bool = False) -> pd.DataFrame:
    """``predict_at`` with a Parquet cache (outputs/cache/preds_<YYYYMMDD>.parquet)."""
    path = config.CACHE_DIR / f"preds_{cutoff:%Y%m%d}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    preds = predict_at(cutoff, get_booster(cutoff, refresh=refresh))[0]
    preds.to_parquet(path, index=False)
    return preds


def reference_images(product_code: str) -> list[Path]:
    """Downloaded reference images for a style (outputs/refs/<code>/*.jpg), sorted."""
    return sorted((config.REFS_DIR / str(product_code).zfill(7)).glob("*.jpg"))


def predict_at(cutoff: date, booster: lgb.Booster | None = None) -> tuple[pd.DataFrame, pd.DataFrame, lgb.Booster]:
    """Predict next-4-week units at ``cutoff``. Returns (preds sorted desc, snapshot, booster).

    preds columns: product_code, pred_units, units_w1, units_w2, units_w4, growth (pred / last 4w - 1),
    y_units (actual, NaN when unknown) + style attributes.
    """
    booster = booster or get_booster(cutoff)
    snap = features.make_snapshot(cutoff)
    _, units = model.predict(booster, "regressor", snap)
    attrs = data.style_attributes(as_of=cutoff)[["product_code", *ATTR_COLS]]
    preds = snap[["product_code", "units_w1", "units_w2", "units_w4", "y_units"]].assign(pred_units=units)
    preds["growth"] = preds["pred_units"] / preds["units_w4"].replace(0, np.nan) - 1
    preds = preds.merge(attrs, on="product_code", how="left")
    for c in ("garment_group_name", "product_group_name", "product_type_name", "index_group_name"):
        preds[c] = preds[c].astype(str)
    return preds.sort_values("pred_units", ascending=False, ignore_index=True), snap, booster


def select_top_k(preds: pd.DataFrame, k: int = 3, diversify_by: str | None = "garment_group_name",
                 min_recent_sales: bool = True) -> pd.DataFrame:
    """Top-``k`` styles by pred_units, at most one per ``diversify_by`` group, optionally requiring
    sales in the last 2 weeks (availability proxy — the dataset has no stock data)."""
    cand = preds.sort_values("pred_units", ascending=False)
    if min_recent_sales:
        cand = cand[cand["units_w2"] > 0]
    if diversify_by:
        cand = cand.drop_duplicates(diversify_by, keep="first")
    out = cand.head(k).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def plot_sales_curve(code: str, cutoff: date, pred_units: float, path: Path, weeks: int = 26) -> None:
    """Weekly units for the last ``weeks`` weeks plus the forecast weekly run-rate."""
    import matplotlib.pyplot as plt

    from data_science import viz

    viz.apply_style()
    sc = data.sales_curve(code, weeks=weeks, end=cutoff)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(sc["week_start"], sc["units"], color=viz.SERIES[0], lw=2, label="actual weekly units")
    fut = pd.date_range(pd.Timestamp(cutoff), periods=config.HORIZON_WEEKS, freq="7D")
    ax.plot(fut, [pred_units / config.HORIZON_WEEKS] * len(fut), color=viz.SERIES[1], lw=2, ls="--",
            label="forecast (avg / week)")
    ax.axvline(pd.Timestamp(cutoff), color=viz.INK2, lw=1)
    ax.set_ylim(0, None)
    ax.set_ylabel("units / week")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Style {code}: last {weeks} weeks + next-4-week forecast", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def write_evidence(winners: pd.DataFrame, cutoff: date, snap: pd.DataFrame, booster: lgb.Booster,
                   out_root: Path = config.EVIDENCE_DIR) -> list[dict]:
    """Write forecast.json + sales_curve.png per winner; return the list of forecast dicts."""
    records = []
    for _, w in winners.iterrows():
        code = w["product_code"]
        d = out_root / code
        d.mkdir(parents=True, exist_ok=True)
        arts = data.top_colour_articles(code, n=3, as_of=cutoff)
        rec = {
            "product_code": code,
            "cutoff": str(cutoff),
            "horizon_weeks": config.HORIZON_WEEKS,
            "rank": int(w["rank"]),
            "predicted_units_next_4w": round(float(w["pred_units"]), 1),
            "units_last_4w": int(w["units_w4"]),
            "units_last_1w": int(w["units_w1"]),
            "naive_run_rate_units": int(4 * w["units_w1"]),  # SHAP drivers are multipliers on this
            "growth_vs_last_4w": round(float(w["growth"]), 3),
            "shap_drivers": model.explain(code, cutoff, snap=snap, booster=booster),
            "attributes": {c: (None if pd.isna(w[c]) else (w[c].item() if hasattr(w[c], "item") else w[c]))
                           for c in ATTR_COLS},
            "representative_articles": arts.to_dict(orient="records"),
            "selection_rule": "top predicted units; max 1 per garment_group_name; sold in last 2 weeks "
                              "(availability proxy — no stock data in the dataset)",
        }
        (d / "forecast.json").write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        plot_sales_curve(code, cutoff, float(w["pred_units"]), d / "sales_curve.png")
        records.append(rec)
    return records


def evidence_for(cutoff: date, codes: list[str], download: bool = True) -> list[dict]:
    """Write forecast.json + sales_curve.png (and fetch reference images) for chosen styles at ``cutoff``.

    ``codes`` are ranked in the given order. Returns [{product_code, forecast_json, sales_curve, ref_images}].
    """
    preds = get_predictions(cutoff)
    codes = [str(c).zfill(7) for c in codes]
    missing = [c for c in codes if c not in set(preds["product_code"])]
    if missing:
        raise ValueError(f"No forecast at {cutoff} for {missing} (they did not sell in the 12 weeks before).")
    winners = preds.set_index("product_code").loc[codes].reset_index()
    winners.insert(0, "rank", np.arange(1, len(winners) + 1))
    recs = write_evidence(winners, cutoff, features.make_snapshot(cutoff), get_booster(cutoff))
    if download:
        download_refs(recs)
    return [{"product_code": r["product_code"],
             "forecast_json": str(config.EVIDENCE_DIR / r["product_code"] / "forecast.json"),
             "sales_curve": str(config.EVIDENCE_DIR / r["product_code"] / "sales_curve.png"),
             "ref_images": [str(p) for p in reference_images(r["product_code"])]} for r in recs]


def _top10_md(preds: pd.DataFrame, title: str) -> str:
    """Markdown table of the top-10 predicted styles (with actuals when known)."""
    top = preds.head(10)
    has_actual = top["y_units"].notna().all()
    actual_rank = preds["y_units"].rank(ascending=False, method="min") if has_actual else None
    hdr = "| # | style | name | type | garment group | pred units | last 4w |" + (" actual | actual rank |" if has_actual else "")
    lines = [f"#### {title}", "", hdr, "|---" * (hdr.count("|") - 1) + "|"]
    for i, r in top.iterrows():
        row = (f"| {i + 1} | {r.product_code} | {r.prod_name} | {r.product_type_name} | {r.garment_group_name} "
               f"| {r.pred_units:,.0f} | {r.units_w4:,.0f} |")
        if has_actual:
            row += f" {r.y_units:,.0f} | {actual_rank[i]:.0f} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def seasonal_comparison(runs: dict[str, tuple[date, pd.DataFrame]], path: Path) -> str:
    """Write the summer-vs-autumn comparison markdown (top-10s + category mix of the top-100)."""
    md = ["# Seasonal comparison: what the model expects to win", "",
          "Same pipeline (retrain on all fully-observed cutoffs, predict next 4 weeks) at two cutoffs. "
          "Category mix = share of predicted units within each run's top-100 styles.", ""]
    mixes = {}
    for name, (c, p) in runs.items():
        md.append(_top10_md(p, f"{name} — cutoff {c} (predicting {c} → {c + timedelta(weeks=4) - timedelta(days=1)})"))
        top100 = p.head(100)
        mixes[name] = top100.groupby("product_group_name")["pred_units"].sum() / top100["pred_units"].sum()
        if p["y_units"].notna().all():
            hits = len(set(p.head(10)["product_code"]) & set(p.nlargest(10, "y_units")["product_code"]))
            md.append(f"Backtest: {hits}/10 of the predicted top-10 were in the actual top-10.\n")
    mix = pd.DataFrame(mixes).fillna(0)
    a, b = mix.columns[:2]
    mix["change (pts)"] = (mix[b] - mix[a]) * 100
    mix = mix.sort_values(b, ascending=False)
    md += ["#### Category mix of the top-100 (product group, share of predicted units)", "",
           f"| product group | {a} | {b} | change (pts) |", "|---|---|---|---|"]
    md += [f"| {g} | {r[a]:.0%} | {r[b]:.0%} | {r['change (pts)']:+.1f} |" for g, r in mix.iterrows()]
    up, down = mix["change (pts)"].idxmax(), mix["change (pts)"].idxmin()
    md += ["", f"**What changed:** from {a} to {b}, {up} gains the most share of the predicted top-100 "
               f"({mix.at[up, 'change (pts)']:+.1f} pts) and {down} loses the most "
               f"({mix.at[down, 'change (pts)']:+.1f} pts)."]
    text = "\n".join(md) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def download_refs(records: list[dict]) -> list[Path]:
    """Download each winner's representative images from Kaggle into outputs/refs/<code>/ (skips existing)."""
    import shutil
    import subprocess
    import sys

    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle.exe"))
    paths = []
    for r in records:
        dest = config.REFS_DIR / r["product_code"]
        for a in r["representative_articles"]:
            p = dest / Path(a["image_path"]).name
            if not p.exists():
                subprocess.run([exe, "competitions", "download", "-c", KAGGLE_COMP, "-f", a["image_path"],
                                "-p", str(dest), "-q"], check=True)
            paths.append(p)
    return paths


def main() -> None:
    """Final top-3 + evidence, seasonal comparison, and reference-image download commands."""
    config.ensure_dirs()
    runs = {}
    for name, c in (("summer 2020", date(2020, 5, 27)), ("autumn 2020", config.FINAL_CUTOFF)):
        # refresh the per-cutoff model + prediction caches (the MCP servers read them)
        preds, snap, booster = predict_at(c, get_booster(c, refresh=True))
        preds.to_parquet(config.CACHE_DIR / f"preds_{c:%Y%m%d}.parquet", index=False)
        runs[name] = (c, preds)  # the autumn (final) run is last, so its snap/booster are kept
        print(f"{name}: {len(preds):,} styles scored at {c}")
    winners = select_top_k(preds)
    recs = write_evidence(winners, config.FINAL_CUTOFF, snap, booster)
    preds.head(50).to_csv(config.OUT_DIR / "predictions_top50.csv", index=False)

    print("\nTop-3 (diversified by garment group):")
    for r in recs:
        a = r["attributes"]
        print(f"  #{r['rank']} {r['product_code']} {a['prod_name']} ({a['product_type_name']}, "
              f"{a['garment_group_name']}): {r['predicted_units_next_4w']:,.0f} units predicted, "
              f"last 4w {r['units_last_4w']:,}")
        for s in r["shap_drivers"][:3]:
            print("      -", s)

    seasonal_comparison(runs, config.FIGURES_DIR / "seasonal_comparison.md")
    print(f"\nSeasonal comparison -> {config.FIGURES_DIR / 'seasonal_comparison.md'}")

    print("\nReference images (3 per winner) and download commands:")
    for r in recs:
        dest = (config.REFS_DIR / r["product_code"]).as_posix()
        for a in r["representative_articles"]:
            print(f"  kaggle competitions download -c {KAGGLE_COMP} -f {a['image_path']} -p {dest}")
    try:
        paths = download_refs(recs)
        print(f"\nDownloaded/verified {len(paths)} reference images in {config.REFS_DIR}")
    except Exception as e:  # noqa: BLE001 — network/credentials; commands above still work by hand
        print(f"\nAutomatic download failed ({e}); run the commands above.")


if __name__ == "__main__":
    main()
