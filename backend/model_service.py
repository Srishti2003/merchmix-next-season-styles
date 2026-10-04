"""Model service: loads outputs/predictions.json once and answers queries. No web-framework code here.

The predictions are produced offline by ``python -m data_science.predict``; the service only reads them, adds
plain-language explanations and turns repo paths into image URLs (``/images/<path under outputs/>``). Images that
are not on disk (reference photos are git-ignored Kaggle data) get a null URL so the frontend can show a placeholder.
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

import config

PREDICTIONS_PATH = config.OUT_DIR / "predictions.json"
IMAGE_PREFIX = "/images"
_REASON = re.compile(r"^(?P<feature>.+?) = (?P<value>.+?) → (?P<direction>raises|lowers) the forecast "
                     r"×(?P<mult>[\d.]+) vs the last-week run-rate$")


class StyleNotFound(LookupError):
    """Unknown or malformed style id (the API maps it to a 404)."""

    def __init__(self, style_id: str, detail: str):
        super().__init__(detail)
        self.style_id, self.detail = style_id, detail


def normalize_style_id(raw: str) -> str:
    """'751471' or '0751471' -> '0751471'. Anything that is not 1-7 digits raises StyleNotFound."""
    s = str(raw).strip()
    if not s.isdigit() or len(s) > 7:
        raise StyleNotFound(s, f"Malformed style_id {s!r}: expected up to 7 digits (a product_code, e.g. 0751471).")
    return s.zfill(7)


def plain_reason(raw: str) -> dict:
    """Turn 'units sold last week = 1,711 → lowers the forecast ×0.62 vs the last-week run-rate' into words."""
    m = _REASON.match(raw)
    if not m:
        return {"text": raw, "feature": None, "value": None, "direction": None, "effect_pct": None, "raw": raw}
    pct = round((float(m["mult"]) - 1) * 100, 1)
    feature = m["feature"]
    text = (f"{feature[0].upper()}{feature[1:]} is {m['value']}: this {m['direction']} the forecast by "
            f"{abs(pct):g}% compared with repeating last week's sales for 4 weeks.")
    return {"text": text, "feature": feature, "value": m["value"], "direction": m["direction"],
            "effect_pct": pct, "raw": raw}


class ModelService:
    def __init__(self, predictions_path: Path = PREDICTIONS_PATH, outputs_dir: Path = config.OUT_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
        self.styles: list[dict] = self.data["styles"]  # top-3 first, then by forecast rank
        self.by_id = {s["style_id"]: s for s in self.styles}
        self.selected = [s for s in self.styles if s["rank"] is not None]

    # --- images -------------------------------------------------------------------------------------------
    def image_url(self, repo_path: str | None) -> str | None:
        """'outputs/refs/x/y.jpg' -> '/images/refs/x/y.jpg' if the file exists, else None."""
        if not repo_path or not repo_path.startswith("outputs/"):
            return None
        rel = repo_path.removeprefix("outputs/")
        return f"{IMAGE_PREFIX}/{rel}" if (self.outputs_dir / rel).is_file() else None

    def _reference_paths(self, s: dict) -> list[str]:
        """Downloaded catalogue photos of the style, best-selling colourway first."""
        code = s["style_id"]
        names = [Path(p).name for p in s["images"]["catalogue"]]
        on_disk = sorted(p.name for p in (self.outputs_dir / "refs" / code).glob("*.jpg"))
        ordered = [n for n in names if n in on_disk] + [n for n in on_disk if n not in names]
        return [f"outputs/refs/{code}/{n}" for n in ordered]

    def primary_image_url(self, s: dict) -> str | None:
        refs = self._reference_paths(s)
        return self.image_url(refs[0]) if refs else None

    # --- queries ------------------------------------------------------------------------------------------
    @cached_property
    def meta(self) -> dict:
        d = self.data
        return {"cutoff": d["cutoff"], "forecast_window": d["forecast_window"], "n_styles": len(self.styles),
                "n_styles_scored": d["n_styles_scored"], "model_version": d["model"]["version"]}

    def get(self, raw_id: str) -> dict:
        code = normalize_style_id(raw_id)
        if code not in self.by_id:
            raise StyleNotFound(code, f"Style {code} is not in the published predictions "
                                      f"(top {len(self.styles)} styles by forecast units).")
        return self.by_id[code]

    def _category(self, s: dict) -> dict:
        return {"product_type": s["category"], "garment_group": s["attributes"].get("garment_group_name")}

    def summary(self, s: dict) -> dict:
        hist = s["sales_history_26w"]
        return {
            "style_id": s["style_id"], "name": s["name"], "rank": s["rank"], "forecast_rank": s["forecast_rank"],
            "prediction_score": s["prediction_score"], "confidence_top1pct": s["confidence_top1pct"],
            "forecast_units": s["forecast_units"], "category": self._category(s),
            "sales_history": {"last_8_weeks": [{"week_start": w["week_start"], "units": w["units"]} for w in hist[-8:]],
                              "units_last_4w": s["performance"]["units_last_4w"]},
            "image_url": self.primary_image_url(s),
        }

    def top(self, limit: int = 10, offset: int = 0) -> dict:
        return {"cutoff": self.meta["cutoff"], "forecast_window": self.meta["forecast_window"],
                "total": len(self.styles), "limit": limit, "offset": offset,
                "styles": [self.summary(s) for s in self.styles[offset:offset + limit]]}

    def why_selected(self, s: dict) -> str:
        w = self.meta["forecast_window"]
        group = s["attributes"].get("garment_group_name")
        if s["rank"] is not None:
            return (f"Selected #{s['rank']}: highest forecast in {group} ({s['forecast_units']:,.0f} units for "
                    f"{w['start']} to {w['end']}), sold in the last 2 weeks; P(top 0.1%) = {s['prediction_score']:.2f}.")
        same = next((x for x in self.selected if x["attributes"].get("garment_group_name") == group), None)
        head = f"Forecast rank #{s['forecast_rank']} ({s['forecast_units']:,.0f} units)"
        if same is not None:
            return (f"{head}; not in the top 3 because {group} is already represented by #{same['rank']} "
                    f"{same['name']} ({same['forecast_units']:,.0f} units forecast).")
        if not s["performance"].get("units_last_2w"):
            return f"{head}; not in the top 3 because it did not sell in the last 2 weeks (availability proxy)."
        third = self.selected[-1]
        return f"{head}; not in the top 3: its forecast is below #3 {third['name']} ({third['forecast_units']:,.0f} units)."

    def concept(self, s: dict) -> dict | None:
        """Generated concept for a top-3 style (from the published evidence files), else None."""
        d = self.outputs_dir / "evidence" / s["style_id"]
        if s["rank"] is None or not (d / "lineage.json").is_file():
            return None
        lineage = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
        final = lineage["final_concept"]
        brief = json.loads((d / "brief.json").read_text(encoding="utf-8")) if (d / "brief.json").is_file() else {}
        critic = {}
        if (d / "critic.jsonl").is_file():
            rows = [json.loads(line) for line in (d / "critic.jsonl").read_text(encoding="utf-8").splitlines() if line]
            critic = next((r for r in reversed(rows) if r.get("concept_path") == final["path"]), {})
        caption = lineage.get("board_caption") or {}
        return {
            "image_url": self.image_url(final["path"]),
            "reference_image_url": self.image_url(final.get("reference_image")),
            "keep": [{"trait": k["trait"], "evidence": k.get("evidence")} for k in brief.get("keep", [])],
            "change": [{"axis": c["axis"], "from": c.get("from"), "to": c.get("to")} for c in brief.get("change", [])],
            "what_changed": caption.get("what_changed"),
            "critic": {"decision": critic.get("decision"), "status": caption.get("status"), "note": critic.get("note"),
                       "max_similarity_to_references": (critic.get("novelty") or {}).get("max_sim_to_refs"),
                       "changes_not_visible": caption.get("brief_changes_not_visible") or []},
        }

    def detail(self, raw_id: str) -> dict:
        s = self.get(raw_id)
        refs = [u for u in (self.image_url(p) for p in self._reference_paths(s)) if u]
        return {
            "style_id": s["style_id"], "name": s["name"], "rank": s["rank"], "forecast_rank": s["forecast_rank"],
            "prediction_score": s["prediction_score"], "confidence_top1pct": s["confidence_top1pct"],
            "p_top0_1pct": s["p_top0_1pct"], "forecast_units": s["forecast_units"], "category": self._category(s),
            "attributes": s["attributes"], "performance": s["performance"],
            "explanation": {"why_selected": self.why_selected(s),
                            "reasons": [plain_reason(r) for r in s["shap_reasons"]], "source": s["shap_source"]},
            "sales_history_26w": s["sales_history_26w"],
            "image_url": refs[0] if refs else None, "reference_image_urls": refs,
            "concept": self.concept(s),
        }

    def health(self) -> dict:
        m = self.meta
        return {"status": "ok", "model_version": m["model_version"], "prediction_cutoff": m["cutoff"],
                "forecast_window": m["forecast_window"], "n_styles": m["n_styles"],
                "n_styles_scored": m["n_styles_scored"]}
