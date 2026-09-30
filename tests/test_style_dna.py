"""The style-dna-brief skill: worked examples in SKILL.md must pass the reusable validator."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skills_lib import style_dna

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / ".claude" / "skills" / "style-dna-brief" / "SKILL.md").read_text(encoding="utf-8")
EXAMPLES = [json.loads(b) for b in re.findall(r"```json\n(\{.*?\})\n```", SKILL, flags=re.S) if '"keep": [\n    {' in b]


def test_frontmatter() -> None:
    """SKILL.md starts with YAML frontmatter holding name + description."""
    fm = SKILL.split("---")[1]
    assert "name: style-dna-brief" in fm and "description:" in fm


@pytest.mark.parametrize("brief", EXAMPLES, ids=[b["product_code"] for b in EXAMPLES])
def test_worked_examples_are_valid(brief: dict) -> None:
    """Each worked example obeys the rules and its prompt is exactly the filled template."""
    fc_path = ROOT / "outputs" / "evidence" / brief["product_code"] / "forecast.json"
    fc = style_dna.load_forecast(fc_path) if fc_path.exists() else None
    assert style_dna.validate_brief(brief, fc) == []
    ptype = fc["attributes"]["product_type_name"] if fc else brief["prompt"].split("(")[1].split(")")[0]
    assert brief["prompt"] == style_dna.build_prompt(ptype, brief["keep"], brief["change"])


def test_validator_catches_violations() -> None:
    """Recolour-only, logos, too few keeps and invisible-only changes are rejected."""
    bad = {"product_code": "0751471", "keep": [{"trait": "slim leg", "evidence": "desc"}],
           "change": [{"axis": "colourway", "from": "black", "to": "navy with a big logo"}],
           "rationale": "x", "prompt": "Redesign this garment (trousers)", "negative_prompt": "x"}
    errs = " | ".join(style_dna.validate_brief(bad))
    for s in ("KEEP must have 3–4", "CHANGE must have 2–3", "recolour", "forbidden"):
        assert s in errs
    invisible = dict(bad, change=[{"axis": "fabric/texture", "from": "a", "to": "b"},
                                  {"axis": "trim/closure", "from": "a", "to": "b"}])
    assert any("thumbnail" in e for e in style_dna.validate_brief(invisible))
