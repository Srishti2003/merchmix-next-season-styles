"""Schema of outputs/predictions.json (written by data_science.predict; read by the backend)."""
from __future__ import annotations

import json

import pytest

import config

PATH = config.OUT_DIR / "predictions.json"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="run python -m data_science.predict first")


@pytest.fixture(scope="module")
def pred() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_top3_is_the_published_selection(pred: dict) -> None:
    top3 = [s for s in pred["styles"] if s["rank"] is not None]
    assert [s["style_id"] for s in top3] == ["0751471", "0762846", "0685814"]
    assert [s["rank"] for s in top3] == [1, 2, 3]
    assert len({s["attributes"]["garment_group_name"] for s in top3}) == 3


def test_every_style_has_the_api_fields(pred: dict) -> None:
    ids = [s["style_id"] for s in pred["styles"]]
    assert len(ids) == len(set(ids))
    for s in pred["styles"]:
        assert len(s["style_id"]) == 7 and s["style_id"].isdigit()
        assert 0 <= s["prediction_score"] <= 1 and 0 <= s["confidence_top1pct"] <= 1
        assert s["forecast_units"] >= 0 and s["category"]
        assert len(s["shap_reasons"]) == 5
        # up to 26 weeks, starting at the first sale for styles launched more recently
        assert len(s["sales_history_26w"]) == min(26, s["performance"]["weeks_since_launch"])
        assert s["sales_history_26w"][-1]["week_start"] < pred["cutoff"]  # history strictly before the cutoff


def test_no_absolute_paths(pred: dict) -> None:
    text = PATH.read_text(encoding="utf-8")
    assert "/workspaces/" not in text and "/home/" not in text and ":\\\\" not in text
    for s in pred["styles"]:
        for p in s["images"]["reference"]:
            assert p.startswith("outputs/refs/")
