"""Reusable core of the style-dna-brief skill (usable without the Agent SDK).

Turns a winning style's evidence (forecast.json) into a KEEP/CHANGE design brief and a FLUX Kontext
image-edit prompt, and validates a brief against the skill's rules. The judgement (which traits to
keep, which changes to make) is done by a person or an LLM; this module supplies the evidence digest,
the allowed change axes, the prompt template and the checks.

CLI:
  python -m skills_lib.style_dna outputs/evidence/0751471/forecast.json            # evidence digest
  python -m skills_lib.style_dna outputs/evidence/0751471/forecast.json brief.json # validate a brief
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHANGE_AXES: tuple[str, ...] = ("silhouette detail", "fabric/texture", "colourway", "trim/closure",
                                "length/proportion", "pattern")
# Axes whose change reads at thumbnail size on a product photo (board columns are ~280 px wide).
THUMBNAIL_VISIBLE: frozenset[str] = frozenset({"colourway", "pattern", "length/proportion", "silhouette detail"})
FORBIDDEN = re.compile(r"\b(logo|logos|brand|branding|h&m|slogan|lettering|text print|monogram|trademark)\b", re.I)

PROMPT_TEMPLATE = ("Redesign this garment ({product_type}) as a new next-season product. "
                   "Keep: {keep}. "
                   "Change: {change}. "
                   "Studio product photo, plain light-grey background, same camera angle, no model, no text.")
NEGATIVE_PROMPT = ("logo, brand name, text, lettering, watermark, human model, mannequin, different garment "
                   "category, copied print, busy background, extra garments")

BRIEF_SCHEMA: dict = {
    "product_code": "str, 7 digits",
    "keep": [{"trait": "str (visual/product trait)", "evidence": "str (attribute / sales / SHAP fact)"}],
    "change": [{"axis": f"one of {list(CHANGE_AXES)}", "from": "str", "to": "str"}],
    "rationale": "str, 1-3 sentences: why these keeps + changes should sell next season",
    "prompt": "str, filled PROMPT_TEMPLATE",
    "negative_prompt": "str (kept for backends that support it; FLUX Kontext [dev] Space ignores it)",
}


def load_forecast(path: str | Path) -> dict:
    """Read a forecast.json written by data_science.select."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence_digest(fc: dict) -> list[str]:
    """Short, quotable evidence lines to tie KEEP traits to (never invents numbers)."""
    a = fc["attributes"]
    colours = [r["colour_group_name"] for r in fc.get("representative_articles", [])]
    lines = [
        f"rank #{fc['rank']} in the diversified top-3: {fc['predicted_units_next_4w']:,.0f} units forecast for the next "
        f"{fc['horizon_weeks']} weeks (last 4 weeks: {fc['units_last_4w']:,})",
        f"product: {a['prod_name']} — {a['product_type_name']} / {a['garment_group_name']} / {a['index_group_name']}",
        f"sells in {a['n_colours']} colours; top colourways: {', '.join(colours) or 'n/a'}",
        f"appearance: {a['graphical_appearance_name']}, main colour {a['colour_group_name']}",
        f"description: {a['detail_desc']}",
    ]
    lines += [f"SHAP: {d}" for d in fc.get("shap_drivers", [])[:3]]
    return lines


def build_prompt(product_type: str, keep: list[dict], change: list[dict]) -> str:
    """Fill the FLUX Kontext prompt template from KEEP traits and CHANGE items."""
    keep_s = ", ".join(k["trait"] for k in keep)
    change_s = ", ".join(c["to"] for c in change)
    return PROMPT_TEMPLATE.format(product_type=product_type.lower(), keep=keep_s, change=change_s)


def make_brief(fc: dict, keep: list[dict], change: list[dict], rationale: str) -> dict:
    """Assemble a brief dict (schema BRIEF_SCHEMA) with the filled prompt."""
    return {"product_code": fc["product_code"], "keep": keep, "change": change, "rationale": rationale,
            "prompt": build_prompt(fc["attributes"]["product_type_name"], keep, change),
            "negative_prompt": NEGATIVE_PROMPT}


def validate_brief(brief: dict, fc: dict | None = None) -> list[str]:
    """Return a list of rule violations (empty = valid). Mirrors the self-check in SKILL.md."""
    errs = []
    keep, change = brief.get("keep", []), brief.get("change", [])
    if not 3 <= len(keep) <= 4:
        errs.append(f"KEEP must have 3–4 traits, has {len(keep)}")
    for k in keep:
        if not k.get("trait") or not k.get("evidence"):
            errs.append(f"KEEP item without trait+evidence: {k}")
    if not 2 <= len(change) <= 3:
        errs.append(f"CHANGE must have 2–3 items, has {len(change)}")
    axes = [c.get("axis") for c in change]
    bad = [x for x in axes if x not in CHANGE_AXES]
    if bad:
        errs.append(f"unknown change axes {bad}; allowed: {list(CHANGE_AXES)}")
    if len(set(axes)) != len(axes):
        errs.append("CHANGE items must use different axes")
    if axes == ["colourway"] or set(axes) <= {"colourway"}:
        errs.append("a colourway-only change is a recolour, not a new product")
    if not THUMBNAIL_VISIBLE & set(axes):
        errs.append(f"at least one change must be visible at thumbnail size: one of {sorted(THUMBNAIL_VISIBLE)}")
    text = " ".join([brief.get("prompt", "")] + [k.get("trait", "") for k in keep] + [c.get("to", "") for c in change])
    if FORBIDDEN.search(text):
        errs.append(f"forbidden brand/logo/text content: {FORBIDDEN.search(text).group(0)!r}")
    if fc:
        ptype = fc["attributes"]["product_type_name"].lower()
        if ptype not in brief.get("prompt", "").lower():
            errs.append(f"prompt must name the product type ({ptype!r}) so the category cannot change")
    for field in ("rationale", "prompt", "negative_prompt"):
        if not brief.get(field):
            errs.append(f"missing {field}")
    return errs


def main(argv: list[str]) -> None:
    """CLI: print the evidence digest + template, or validate a brief JSON."""
    fc = load_forecast(argv[0])
    if len(argv) == 1:
        print("\n".join(f"- {s}" for s in evidence_digest(fc)))
        print(f"\nallowed change axes: {', '.join(CHANGE_AXES)}\ntemplate: {PROMPT_TEMPLATE}")
        return
    brief = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    errs = validate_brief(brief, fc)
    print("VALID" if not errs else "INVALID:\n" + "\n".join(f"- {e}" for e in errs))


if __name__ == "__main__":
    main(sys.argv[1:])
