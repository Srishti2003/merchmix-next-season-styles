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
    parts = [e.value for e in (*at.markdown, *at.caption, *at.header, *at.subheader)]
    return "\n".join(str(p) for p in parts)


def test_top_page_renders(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.header[0].value == "Top predicted styles"
    text = _text(at)
    assert "#1 Pluto RW slacks (1)" in text and "#3 RICHIE HOOD" in text
    assert "Critic: not approved" in text  # RICHIE's concept is shown honestly
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 50


def test_detail_page_renders_for_top3(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        at = AppTest.from_file(APP, default_timeout=60)
        at.query_params["page"] = "detail"
        at.query_params["style_id"] = "751471"
        at.run()
    assert not at.exception
    text = _text(at)
    assert at.header[0].value.startswith("Pluto RW slacks (1)")
    assert "Selected #1" in text and "KEEP" in text and "CHANGE" in text and "Critic: approved" in text
    assert sum(1 for m in at.markdown if m.value.startswith(("▲", "▼"))) == 5


def test_seasonal_and_performance_pages_render(api) -> None:
    with mock.patch("httpx.get", side_effect=_mock_get(api)):
        for page, expect in (("seasonal", "7/10"), ("performance", "Regressor vs baselines")):
            at = AppTest.from_file(APP, default_timeout=60)
            at.query_params["page"] = page
            at.run()
            assert not at.exception, page
            assert expect in _text(at), page


def test_api_down_shows_message() -> None:
    def down(*_, **__):
        raise httpx.ConnectError("connection refused")
    with mock.patch("httpx.get", side_effect=down):
        at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "The API is not reachable" in at.error[0].value
    assert any("uvicorn backend.api:app" in c.value for c in at.code)
