"""Streamlit AppTest smoke tests for frontend/app.py, with the API mocked by the real FastAPI app (TestClient)."""
from __future__ import annotations

from unittest import mock

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config

pytestmark = pytest.mark.skipif(not (config.OUT_DIR / "predictions.json").exists(),
                                reason="run python -m data_science.predict first")
APP = str(config.ROOT / "frontend" / "app.py")


@pytest.fixture(scope="module")
def api():
    from fastapi.testclient import TestClient

    from backend.api import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _mock_get(api):
    """httpx.get replacement that routes the frontend's requests to the in-process API."""
    def get(url: str, params=None, timeout=None):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        r = api.get("/" + path, params=params)
        return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))
    return get


def _text(at: AppTest) -> str:
    parts = [e.value for e in (*at.markdown, *at.caption, *at.header, *at.subheader, *at.info, *at.success)]
    return "\n".join(str(p) for p in parts)


def _run(api, **query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in query.items():
        at.query_params[k] = v
    return at.run()


def test_overview_is_the_landing_page(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api)
        assert not at.exception
        assert at.header[0].value == "Overview"
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Styles scored"] == "20,318"
        assert metrics["Forecast window"] == "4 weeks"
        assert metrics["Beats last week × 4"] == "7/10"
        assert all(m.help for m in at.metric)  # every metric has a tooltip
        text = _text(at)
        assert "How to read this" in at.info[0].value and "Pick #3 · rank 4 by units" in text
        assert any(getattr(e, "label", None) == "Glossary" for e in at.sidebar.children.values())  # icon expander
        assert "NDCG@50" in text and "PR-AUC" in text and "Calibration / ECE" in text  # glossary entries
        at.button(key="ov_0685814").click().run()  # winner card -> detail
        assert not at.exception
        assert at.header[0].value == "RICHIE HOOD  ·  0685814"
        at.button(key="back").click().run()
        assert at.header[0].value == "Top predicted styles"


def test_top_page_renders(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api, page="top")
    assert not at.exception
    assert at.header[0].value == "Top predicted styles"
    text = _text(at)
    assert "Pluto RW slacks" in text and "(1)" not in text and "RICHIE HOOD" in text
    assert "Pick #1 · rank 1 by units" in text and "Pick #3 · rank 4 by units" in text
    assert "Critic: not approved" in text  # RICHIE's concept is shown honestly
    assert "Trousers · Trousers" not in text and "Hoodie · Jersey Basic" in text
    assert ("so Jade HW Skinny Denim TRS (#2 by units) is skipped because Trousers is already covered by "
            "Pluto RW slacks.") in text
    table = at.dataframe[0].value
    assert len(at.dataframe) == 1 and len(table) == 50
    assert list(table["Rank"][:4]) == [1, 2, 3, 4]
    assert list(table["Top-3 pick"][:4]) == ["✓", "", "✓", "✓"]
    assert "Chance of top 1%" in table.columns and "Index group" in table.columns


def test_top_page_search_filters_and_sort(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api, page="top")
        at.text_input(key="q_AW2020").input("pluto").run()
        assert list(at.dataframe[0].value["Name"]) == ["Pluto RW slacks"]
        at.text_input(key="q_AW2020").input("685814").run()  # style id without the leading zero
        assert list(at.dataframe[0].value["Style"]) == ["0685814"]
        at.text_input(key="q_AW2020").input("").run()
        at.toggle(key="top3_AW2020").set_value(True).run()
        assert list(at.dataframe[0].value["Style"]) == ["0751471", "0762846", "0685814"]
        at.toggle(key="top3_AW2020").set_value(False).run()
        group = at.multiselect(key="fi_AW2020").options[-1]
        at.multiselect(key="fi_AW2020").set_value([group]).run()
        table = at.dataframe[0].value
        assert len(table) and set(table["Index group"]) == {group}
        at.multiselect(key="fi_AW2020").set_value([]).run()
        at.selectbox(key="sort_AW2020").set_value("Prediction score").run()
        scores = list(at.dataframe[0].value["Prediction score"])
        assert scores == sorted(scores, reverse=True)
        at.text_input(key="q_AW2020").input("no such style").run()
        assert not at.exception and len(at.dataframe) == 0
        assert "No styles match" in at.info[0].value


def test_detail_page_renders_for_top3(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api, page="detail", style_id="751471")
        assert not at.exception
        text = _text(at)
        assert at.header[0].value == "Pluto RW slacks  ·  0751471"
        assert "Selected #1" in text and "KEEP" in text and "CHANGE" in text and "Critic: approved" in text
        reasons = next(m.value for m in at.markdown if ":green[▲]" in m.value or ":red[▼]" in m.value)
        assert len(reasons.splitlines()) == 5 and all(line.startswith("- :") for line in reasons.splitlines())
        assert "Big recent weeks are partly discounted" in text
        assert at.button(key="prev").disabled
        at.button(key="next").click().run()  # next by forecast rank
        assert at.header[0].value == "Jade HW Skinny Denim TRS  ·  0706016"
        at.button(key="prev").click().run()
        assert at.header[0].value == "Pluto RW slacks  ·  0751471"


def test_ss2020_detail_shows_actuals(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api, page="detail", style_id="0854677", season="SS2020")
    assert not at.exception
    metrics = {m.label: m for m in at.metric}
    assert "Actual units" in metrics and "vs forecast" in metrics["Actual units"].delta


def test_seasonal_and_performance_pages_render(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = _run(api, page="seasonal")
        assert not at.exception
        text = _text(at)
        assert "7 of the predicted top 10 were in the actual top 10" in at.success[0].value
        assert "7/10" in text and "Risk flag" in text and "C Lolly Top" in text
        assert "In actual top 10" in at.dataframe[0].value.columns
        at = _run(api, page="performance")
        assert not at.exception
        text = _text(at)
        assert "What this means for a merchandiser" in at.info[0].value
        assert "@12 follows the H&M Kaggle competition's MAP@12 convention" in text
        metrics = {m.label: m for m in at.metric}
        assert metrics["Ranking quality (NDCG@50)"].value == "0.924"
        assert metrics["Ranking quality (NDCG@50)"].delta == "+0.018 vs last week × 4"
        assert metrics["Winner classifier PR-AUC"].value == "0.807"
        assert metrics["Backtest weeks won"].value == "7/10"
        assert [e.label for e in at.main.expander].count("Show numbers") == 3
        assert len(at.dataframe) == 3  # the exact tables, inside the expanders


def test_api_down_shows_message() -> None:
    def down(*_, **__):
        raise httpx.ConnectError("connection refused")
    with mock.patch("httpx.get", side_effect=down):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "The API is not reachable" in at.error[0].value
    assert any("uvicorn backend.api:app" in c.value for c in at.code)
