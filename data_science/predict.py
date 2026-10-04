"""Write outputs/predictions.json: the ranked styles at the final cutoff, ready for the API and the frontend.

  python -m data_science.predict            # top 200 styles by forecast units (default)
  python -m data_science.predict --n 500

Inputs (nothing is retrained and no published file is rewritten):
- outputs/classifier/scores_final.parquet (data_science.classify --combine): forecast units from the published final
  regressor, calibrated classifier probabilities, the top-3 rank;
- models/lgbm_final_20200923.txt (published final regressor) for the SHAP reasons of styles without evidence;
- outputs/evidence/<code>/{forecast.json, lineage.json} for the 3 winners (published SHAP drivers, final concept);
- the weekly style cache for the 26-week sales history.

Per style: style_id (= product_code), rank (1-3 for the selected winners, else null), forecast_rank (position by
forecast units among all scored styles), prediction_score, confidence_top1pct, p_top0_1pct, forecast_units,
attributes, recent performance, shap_reasons, sales_history_26w and image paths (relative to the repo root).
"""
from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd

import config
from data_science import classify, data, features, model, select

OUT_PATH = config.OUT_DIR / "predictions.json"
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
    return {
        "reference": [_rel(p) for p in select.reference_images(code)],  # local only (Kaggle photos, not committed)
        "catalogue": catalogue,                                         # Kaggle paths of the top-colour articles
        "concept": concept["path"] if concept else None,
        "concept_reference": concept.get("reference_image") if concept else None,
        "sales_curve": _rel(config.EVIDENCE_DIR / code / "sales_curve.png")
        if (config.EVIDENCE_DIR / code / "sales_curve.png").exists() else None,
    }


def _model_info() -> dict:
    """Model files behind the scores, and a short version string for the API's /health."""
    reg = json.loads(model.META_PATH.read_text())
    clf = {s: json.loads(classify.paths(s)["meta"].read_text()) for s in (classify.WINNER_SHARE, classify.STRICT_SHARE)}
    files = {"regressor": _rel(config.MODELS_DIR / f"lgbm_final_{config.FINAL_CUTOFF:%Y%m%d}.txt"),
             **{f"classifier_top{s * 100:g}pct": _rel(classify.paths(s)["model"]) for s in clf},
             **{f"calibrator_top{s * 100:g}pct": _rel(classify.paths(s)["calibrator"]) for s in clf}}
    version = (f"regressor-{config.FINAL_CUTOFF:%Y%m%d}-r{reg['best_iterations']['regressor']}"
               f"+clf-top1pct-r{clf[classify.WINNER_SHARE]['rounds']}"
               f"+clf-top0.1pct-r{clf[classify.STRICT_SHARE]['rounds']}")
    return {"version": version, "files": files}


def build(n: int = 200) -> dict:
    cutoff = config.FINAL_CUTOFF
    scores = pd.read_parquet(SCORES_PATH).sort_values("forecast_units", ascending=False, ignore_index=True)
    scores["forecast_rank"] = range(1, len(scores) + 1)
    top = scores[(scores["forecast_rank"] <= n) | scores["rank"].notna()]
    snap = features.make_snapshot(cutoff).set_index("product_code")
    attrs = data.style_attributes(as_of=cutoff).set_index("product_code")
    booster = select.get_booster(cutoff)
    snap_reset = snap.reset_index()

    styles = []
    for r in top.itertuples():
        code = r.product_code
        fc, ln = _evidence(code)
        s, a = snap.loc[code], attrs.loc[code]
        if fc:  # the published winners keep exactly the drivers shown on the evidence sheet
            reasons, source = fc["shap_drivers"], f"outputs/evidence/{code}/forecast.json"
            catalogue = [x["image_path"] for x in fc["representative_articles"]]
        else:
            reasons = model.explain(code, cutoff, snap=snap_reset, booster=booster)
            source = "models/lgbm_final_20200923.txt (TreeSHAP)"
            catalogue = data.top_colour_articles(code, n=3, as_of=cutoff)["image_path"].tolist()
        hist = data.sales_curve(code, weeks=HISTORY_WEEKS, end=cutoff)
        styles.append({
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
        })
    styles.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, x["forecast_rank"]))
    src = scores["score_source"].iat[0]
    return {
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "forecast_window": {"start": f"{cutoff:%Y-%m-%d}",
                            "end": f"{cutoff + timedelta(weeks=config.HORIZON_WEEKS) - timedelta(days=1):%Y-%m-%d}"},
        "n_styles_scored": int(len(scores)),
        "model": _model_info(),
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


def main(n: int = 200) -> Path:
    out = build(n)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    top3 = [s for s in out["styles"] if s["rank"]]
    print(f"wrote {_rel(OUT_PATH)}: {len(out['styles'])} styles")
    for s in top3:
        print(f"  #{s['rank']} {s['style_id']} {s['name']}: prediction_score {s['prediction_score']}, "
              f"{s['forecast_units']:,} units, concept {s['images']['concept']}")
    return OUT_PATH


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write outputs/predictions.json")
    ap.add_argument("--n", type=int, default=200, help="number of styles by forecast units (top-3 always included)")
    main(ap.parse_args().n)
