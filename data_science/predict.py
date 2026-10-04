"""Write the prediction files the API and frontend read: ranked styles per season, nothing retrained.

  python -m data_science.predict                    # AW2020: outputs/predictions.json (cutoff 2020-09-23)
  python -m data_science.predict --season SS2020    # SS2020: outputs/predictions_SS2020.json (cutoff 2020-05-27)
  python -m data_science.predict --n 500            # more styles (default: top 200 by forecast units)

AW2020 (the forecast; the target window 2020-09-23 → 2020-10-20 is after the data):
- outputs/classifier/scores_final.parquet (data_science.classify --combine): forecast units from the published final
  regressor models/lgbm_final_20200923.txt, calibrated classifier probabilities, the top-3 rank;
- outputs/evidence/<code>/{forecast.json, lineage.json} for the 3 winners (published SHAP drivers, final concept).
SS2020 (a backtest season; the window 2020-05-27 → 2020-06-23 is observed, so actual units are included):
- forecast units from the published summer regressor models/lgbm_final_20200527.txt (via select.get_predictions);
- classifier probabilities = the out-of-fold backtest predictions at 2020-05-27 (models trained on cutoffs up to
  2020-04-29, isotonic maps fitted on earlier folds only), so no information from the window is used;
- no concepts or new images.

Per style: style_id (= product_code), rank (1-3 for the selected winners, else null), forecast_rank (position by
forecast units among all scored styles), prediction_score, confidence_top1pct, p_top0_1pct, forecast_units,
attributes, recent performance, shap_reasons, sales_history_26w, image paths (relative to the repo root) and, for
observed seasons, actual units and rank.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
from data_science import classify, data, features, model, select

SEASONS: dict[str, dict] = {
    "AW2020": {"label": "Autumn/Winter 2020 (forecast)", "cutoff": config.FINAL_CUTOFF,
               "path": config.OUT_DIR / "predictions.json"},
    "SS2020": {"label": "Spring/Summer 2020 (backtest)", "cutoff": date(2020, 5, 27),
               "path": config.OUT_DIR / "predictions_SS2020.json"},
}
DEFAULT_SEASON = "AW2020"
OUT_PATH = SEASONS[DEFAULT_SEASON]["path"]
SCORES_PATH = classify.OUT_DIR / "scores_final.parquet"
HISTORY_WEEKS = 26


def _rel(p: Path | str) -> str:
    """Repo-relative POSIX path (never an absolute path in the published JSON)."""
    p = Path(p)
    return (p.resolve().relative_to(config.ROOT) if p.is_absolute() else p).as_posix()


def _num(v, nd: int = 4):
    """JSON-safe number: None for NaN, int for whole counts, rounded float otherwise."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    f = float(v)
    return int(f) if f.is_integer() else round(f, nd)


def _prob(v):
    """Probability as a float rounded to 4 decimals (1.0 stays 1.0), None for NaN."""
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _str(v):
    return None if v is None or pd.isna(v) else str(v)


def _evidence(code: str) -> tuple[dict | None, dict | None]:
    d = config.EVIDENCE_DIR / code
    fc, ln = d / "forecast.json", d / "lineage.json"
    return (json.loads(fc.read_text(encoding="utf-8")) if fc.exists() else None,
            json.loads(ln.read_text(encoding="utf-8")) if ln.exists() else None)


def _images(code: str, lineage: dict | None, catalogue: list[str]) -> dict:
    concept = lineage["final_concept"] if lineage else None
    curve = config.EVIDENCE_DIR / code / "sales_curve.png"
    return {
        "reference": [_rel(p) for p in select.reference_images(code)],  # local only (Kaggle photos, not committed)
        "catalogue": catalogue,                                         # Kaggle paths of the top-colour articles
        "concept": concept["path"] if concept else None,
        "concept_reference": concept.get("reference_image") if concept else None,
        "sales_curve": _rel(curve) if lineage and curve.exists() else None,
    }


def _model_info(cutoff: date) -> dict:
    """Model files behind the scores, and a short version string for the API's /health."""
    reg = json.loads(model.META_PATH.read_text())
    clf = {s: json.loads(classify.paths(s)["meta"].read_text()) for s in (classify.WINNER_SHARE, classify.STRICT_SHARE)}
    reg_file = config.MODELS_DIR / f"lgbm_final_{cutoff:%Y%m%d}.txt"
    version = (f"regressor-{cutoff:%Y%m%d}-r{reg['best_iterations']['regressor']}"
               f"+clf-top1pct-r{clf[classify.WINNER_SHARE]['rounds']}"
               f"+clf-top0.1pct-r{clf[classify.STRICT_SHARE]['rounds']}")
    if cutoff == config.FINAL_CUTOFF:
        files = {"regressor": _rel(reg_file),
                 **{f"classifier_top{s * 100:g}pct": _rel(classify.paths(s)["model"]) for s in clf},
                 **{f"calibrator_top{s * 100:g}pct": _rel(classify.paths(s)["calibrator"]) for s in clf}}
    else:
        version += "-oof"
        files = {"regressor": _rel(reg_file),
                 **{f"classifier_top{s * 100:g}pct": f"out-of-fold backtest prediction ({_rel(classify.paths(s)['oof'])})"
                    for s in clf}}
    return {"version": version, "files": files}


def _final_scores() -> pd.DataFrame:
    """AW2020: the combined final scores (regressor units + calibrated classifier probabilities + top-3 rank)."""
    return pd.read_parquet(SCORES_PATH)


def _backtest_scores(cutoff: date) -> pd.DataFrame:
    """Observed season: published per-cutoff regressor forecast + out-of-fold calibrated classifier probabilities."""
    c = pd.Timestamp(cutoff)
    preds = select.get_predictions(cutoff)
    df = preds[["product_code", "pred_units", "units_w2", "garment_group_name", "y_units"]].rename(
        columns={"pred_units": "forecast_units"})
    for share, col in ((classify.WINNER_SHARE, "confidence_top1pct"), (classify.STRICT_SHARE, "p_strict")):
        oof = pd.read_parquet(classify.paths(share)["oof"])
        oof = oof[(oof["cutoff"] == c) & oof["p_cal"].notna()][["product_code", "p_cal"]]
        if oof.empty:
            raise ValueError(f"No calibrated out-of-fold classifier predictions at {cutoff}.")
        df = df.merge(oof.rename(columns={"p_cal": col}), on="product_code", how="left")
    df["prediction_score"] = df["p_strict"]
    df["score_source"] = f"calibrated P(top {classify.STRICT_SHARE * 100:g}%)"
    df = df.sort_values("forecast_units", ascending=False, ignore_index=True)
    top3 = classify.select_top_k(df)
    df["rank"] = df["product_code"].map(dict(zip(top3["product_code"], top3["rank"])))
    df["actual_rank"] = df["y_units"].rank(method="min", ascending=False)
    return df


def build(season: str = DEFAULT_SEASON, n: int = 200) -> dict:
    spec = SEASONS[season]
    cutoff = spec["cutoff"]
    final = cutoff == config.FINAL_CUTOFF
    scores = _final_scores() if final else _backtest_scores(cutoff)
    scores = scores.sort_values("forecast_units", ascending=False, ignore_index=True)
    scores["forecast_rank"] = range(1, len(scores) + 1)
    top = scores[(scores["forecast_rank"] <= n) | scores["rank"].notna()]
    snap = features.make_snapshot(cutoff).set_index("product_code")
    attrs = data.style_attributes(as_of=cutoff).set_index("product_code")
    booster = select.get_booster(cutoff)
    snap_reset = snap.reset_index()
    window_end = cutoff + timedelta(weeks=config.HORIZON_WEEKS)

    styles = []
    for r in top.itertuples():
        code = r.product_code
        fc, ln = _evidence(code) if final else (None, None)  # evidence and concepts exist for AW2020 only
        s, a = snap.loc[code], attrs.loc[code]
        if fc:  # the published winners keep exactly the drivers shown on the evidence sheet
            reasons, source = fc["shap_drivers"], f"outputs/evidence/{code}/forecast.json"
            catalogue = [x["image_path"] for x in fc["representative_articles"]]
        else:
            reasons = model.explain(code, cutoff, snap=snap_reset, booster=booster)
            source = f"models/lgbm_final_{cutoff:%Y%m%d}.txt (TreeSHAP)"
            catalogue = data.top_colour_articles(code, n=3, as_of=cutoff)["image_path"].tolist()
        hist = data.sales_curve(code, weeks=HISTORY_WEEKS, end=cutoff)
        rec = {
            "style_id": code,
            "rank": None if pd.isna(r.rank) else int(r.rank),
            "forecast_rank": int(r.forecast_rank),
            "prediction_score": _prob(r.prediction_score),
            "confidence_top1pct": _prob(r.confidence_top1pct),
            "p_top0_1pct": _prob(r.p_strict),
            "forecast_units": _num(round(r.forecast_units, 1)),
            "name": _str(a["prod_name"]),
            "category": _str(a["product_type_name"]),
            "attributes": {c: _str(a[c]) for c in ("product_group_name", "garment_group_name", "index_group_name",
                                                   "section_name", "colour_group_name", "graphical_appearance_name",
                                                   "detail_desc")} | {"n_colours": _num(a["n_colours"])},
            "performance": {
                "units_last_1w": _num(s["units_w1"]), "units_last_2w": _num(s["units_w2"]),
                "units_last_4w": _num(s["units_w4"]), "units_last_12w": _num(s["units_w12"]),
                "units_same_4w_last_year": _num(s["ly_units_h"]), "units_to_date": _num(a["units_total"]),
                "buyers_last_4w": _num(s["buyers_w4"]), "weeks_since_launch": _num(s["weeks_since_launch"]),
                "discount_vs_peak_price": _num(s["discount"]), "online_share_last_4w": _num(s["online_share_w4"]),
            },
            "shap_reasons": reasons,
            "shap_source": source,
            "sales_history_26w": [{"week_start": f"{w:%Y-%m-%d}", "units": int(u), "buyers": int(b)}
                                  for w, u, b in zip(hist["week_start"], hist["units"], hist["buyers"])],
            "images": _images(code, ln, catalogue),
            "board_caption": ln.get("board_caption") if ln else None,
        }
        if not final:  # observed window: actual units, rank and the 4 weekly values
            fut = data.sales_curve(code, weeks=config.HORIZON_WEEKS, end=window_end)
            fut = fut[fut["week_start"] >= pd.Timestamp(cutoff)]
            rec["actual"] = {"units": _num(r.y_units), "rank": _num(r.actual_rank),
                             "weekly": [{"week_start": f"{w:%Y-%m-%d}", "units": int(u)}
                                        for w, u in zip(fut["week_start"], fut["units"])]}
        styles.append(rec)
    styles.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, x["forecast_rank"]))
    src = scores["score_source"].iat[0]
    out = {
        "season": season,
        "season_label": spec["label"],
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "forecast_window": {"start": f"{cutoff:%Y-%m-%d}", "end": f"{window_end - timedelta(days=1):%Y-%m-%d}"},
        "observed": not final,
        "has_concepts": final,
        "n_styles_scored": int(len(scores)),
        "model": _model_info(cutoff),
        "selection_rule": "top-3 = highest forecast units, at most one style per garment group, sold in the last "
                          "2 weeks (availability proxy; no stock data)",
        "definitions": {
            "style_id": "product_code (7 characters) = article_id // 1000; all colour variants of one design",
            "forecast_units": "regressor forecast of units sold in the forecast window",
            "prediction_score": f"{src}: probability that the style's next-4-week units rank in that top share "
                                "of active styles. A relative-strength signal; the ranking is forecast_units",
            "confidence_top1pct": "calibrated probability of being a top-1% style over the next 4 weeks",
            "p_top0_1pct": "calibrated probability of being a top-0.1% style over the next 4 weeks",
            "shap_reasons": "top-5 TreeSHAP drivers of the regressor, as multipliers on the last-week x 4 run-rate",
            "sales_history_26w": "weekly units and buyers for the last 26 weeks before the cutoff (from the first "
                                 "sale for styles launched more recently; zero weeks included)",
            "images.reference": "local Kaggle photos (outputs/refs/, not committed); empty if not downloaded",
            "images.catalogue": "Kaggle image paths of the best-selling colour articles",
        },
        "styles": styles,
    }
    if not final:
        out["definitions"]["actual"] = ("observed units in the forecast window, the style's rank by them among all "
                                        "scored styles, and the 4 weekly values")
    return out


def main(season: str = DEFAULT_SEASON, n: int = 200) -> Path:
    out = build(season, n)
    path = SEASONS[season]["path"]
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {_rel(path)}: {season}, {len(out['styles'])} styles, cutoff {out['cutoff']}")
    for s in (x for x in out["styles"] if x["rank"]):
        actual = f", actual {s['actual']['units']:,} (rank {s['actual']['rank']})" if "actual" in s else ""
        print(f"  #{s['rank']} {s['style_id']} {s['name']}: prediction_score {s['prediction_score']}, "
              f"{s['forecast_units']:,} units{actual}")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write the per-season prediction files")
    ap.add_argument("--season", choices=list(SEASONS), default=DEFAULT_SEASON)
    ap.add_argument("--n", type=int, default=200, help="number of styles by forecast units (top-3 always included)")
    a = ap.parse_args()
    main(a.season, a.n)
