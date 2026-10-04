"""Backend API (FastAPI TestClient over the committed outputs/predictions.json)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config

pytestmark = pytest.mark.skipif(not (config.OUT_DIR / "predictions.json").exists(),
                                reason="run python -m data_science.predict first")


@pytest.fixture(scope="module")
def client() -> TestClient:
    from backend.api import app
    return TestClient(app)


def test_top_list(client: TestClient) -> None:
    r = client.get("/styles/top", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 200 and body["limit"] == 5 and len(body["styles"]) == 5
    assert [s["style_id"] for s in body["styles"][:3]] == ["0751471", "0762846", "0685814"]
    assert [s["rank"] for s in body["styles"]] == [1, 2, 3, None, None]
    first = body["styles"][0]
    assert set(first) >= {"style_id", "name", "prediction_score", "confidence_top1pct", "forecast_units", "rank",
                          "category", "sales_history", "image_url"}
    assert first["category"] == {"product_type": "Trousers", "garment_group": "Trousers"}
    assert len(first["sales_history"]["last_8_weeks"]) == 8
    assert first["sales_history"]["units_last_4w"] == 9182
    assert first["image_url"] is None or first["image_url"].startswith("/images/refs/0751471/")


def test_offset_paginates(client: TestClient) -> None:
    a = client.get("/styles/top", params={"limit": 4}).json()["styles"]
    b = client.get("/styles/top", params={"limit": 2, "offset": 2}).json()["styles"]
    assert [s["style_id"] for s in b] == [s["style_id"] for s in a[2:4]]
    assert len(client.get("/styles/top", params={"limit": 200}).json()["styles"]) == 200


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"limit": "ten"}])
def test_limit_bounds_return_422(client: TestClient, params: dict) -> None:
    r = client.get("/styles/top", params=params)
    assert r.status_code == 422
    assert r.json()["detail"] == "Invalid request parameters." and r.json()["errors"]


def test_detail_top3_has_concept(client: TestClient) -> None:
    r = client.get("/styles/0685814")
    assert r.status_code == 200
    d = r.json()
    assert d["rank"] == 3 and d["category"]["garment_group"] == "Jersey Basic"
    assert len(d["explanation"]["reasons"]) == 5
    assert all(x["text"] and x["direction"] in ("raises", "lowers") for x in d["explanation"]["reasons"])
    assert d["explanation"]["why_selected"].startswith("Selected #3")
    assert len(d["sales_history_26w"]) == 26
    c = d["concept"]
    assert c["image_url"] == "/images/evidence/0685814/concept_2.png"
    assert c["keep"] and c["change"] and "from" in c["change"][0]
    assert c["critic"]["decision"] == "revise" and c["critic"]["status"] == "Critic: not approved"


def test_detail_without_concept(client: TestClient) -> None:
    d = client.get("/styles/0706016").json()  # Jade: forecast rank 2, same garment group as Pluto
    assert d["rank"] is None and d["concept"] is None
    assert "already represented by #1" in d["explanation"]["why_selected"]


@pytest.mark.parametrize("bad", ["0000001", "abc", "12345678", "07514 71"])
def test_unknown_or_malformed_id_is_404_json(client: TestClient, bad: str) -> None:
    r = client.get(f"/styles/{bad}")
    assert r.status_code == 404
    body = r.json()
    assert set(body) == {"detail", "style_id"} and body["detail"]


def test_leading_zero_is_optional(client: TestClient) -> None:
    a, b = client.get("/styles/751471"), client.get("/styles/0751471")
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json() and a.json()["style_id"] == "0751471"


def test_health(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["prediction_cutoff"] == "2020-09-23"
    assert h["n_styles"] == 200 and h["model_version"].startswith("regressor-20200923")


def test_images_served_only_from_allowed_places(client: TestClient) -> None:
    assert client.get("/images/evidence/0751471/concept_2.png").headers["content-type"] == "image/png"
    assert client.get("/images/generated_concepts.png").status_code == 200
    for bad in ("/images/cache/train.parquet", "/images/predictions.json", "/images/evidence/0751471/brief.json",
                "/images/refs/0751471/missing.jpg"):
        r = client.get(bad)
        assert r.status_code == 404 and "detail" in r.json()


def test_cors_allows_streamlit_origin(client: TestClient) -> None:
    r = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"
