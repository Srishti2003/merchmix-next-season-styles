"""Backend API (FastAPI TestClient over the committed outputs/predictions.json)."""
from __future__ import annotations

import io

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
    assert first["category"] == {"product_type": "Trousers", "garment_group": "Trousers", "index_group": "Ladieswear"}
    assert body["n_styles_scored"] == 20318
    assert first["name"] == "Pluto RW slacks" and first["raw_name"] == "Pluto RW slacks (1)"
    assert len(first["sales_history"]["last_8_weeks"]) == 8
    assert first["sales_history"]["units_last_4w"] == 9182
    assert first["image_url"].startswith(("/images/refs/0751471/", "/images/board-ref/0751471.png"))


@pytest.mark.parametrize("raw, shown", [("Pluto RW slacks (1)", "Pluto RW slacks"), ("Edda top(1)", "Edda top"),
                                         ("Lucy blouse", "Lucy blouse"), ("Rose thong 7-pack(2)", "Rose thong 7-pack"),
                                         (None, None)])
def test_display_name_strips_copy_number(raw, shown) -> None:
    from backend.model_service import display_name
    assert display_name(raw) == shown


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


def test_board_reference_photo(client: TestClient) -> None:
    from PIL import Image

    from backend import model_service as ms
    from image import board
    assert (ms.BOARD_MARGIN, ms.BOARD_GUTTER, ms.BOARD_COL_W, ms.BOARD_IMG_W, ms.BOARD_IMG_H) == \
        (board.MARGIN, board.GUTTER, board.COL_W, board.IMG_W, board.IMG_H)
    for code in ("0751471", "0762846", "0685814"):
        r = client.get(f"/images/board-ref/{code}.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(r.content)).size == (board.IMG_W - 4, board.IMG_H - 4)
    for bad in ("0706016", "abc"):  # not a board winner / malformed
        assert client.get(f"/images/board-ref/{bad}.png").status_code == 404


def test_photo_fallback_without_refs(tmp_path) -> None:
    """Hosted demo (no outputs/refs/): winners get the board crop, everyone else a null URL for the placeholder."""
    import shutil

    from backend.model_service import ModelService
    shutil.copy(config.OUT_DIR / "generated_concepts.png", tmp_path)
    shutil.copytree(config.OUT_DIR / "evidence", tmp_path / "evidence")
    svc = ModelService(config.OUT_DIR / "predictions.json", tmp_path)
    assert svc.primary_image_url(svc.get("0751471")) == "/images/board-ref/0751471.png"
    assert svc.detail("0762846")["reference_image_urls"] == ["/images/board-ref/0762846.png"]
    assert svc.detail("0685814")["concept"]["reference_image_url"] == "/images/board-ref/0685814.png"
    assert svc.detail("0706016")["image_url"] is None


@pytest.mark.skipif(not (config.REFS_DIR / "0751471").is_dir(), reason="catalogue photos not downloaded")
def test_downloaded_photos_take_priority(client: TestClient) -> None:
    d = client.get("/styles/0751471").json()
    assert d["image_url"].startswith("/images/refs/0751471/")
    assert not any(u.startswith("/images/board-ref/") for u in d["reference_image_urls"])


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


def test_seasons(client: TestClient) -> None:
    s = client.get("/seasons").json()["seasons"]
    by_id = {x["id"]: x for x in s}
    assert set(by_id) == {"AW2020", "SS2020"}
    assert by_id["AW2020"]["default"] and not by_id["AW2020"]["observed"] and by_id["AW2020"]["has_concepts"]
    assert by_id["SS2020"]["observed"] and not by_id["SS2020"]["has_concepts"]
    assert by_id["SS2020"]["cutoff"] == "2020-05-27"


def test_ss2020_top_has_actuals(client: TestClient) -> None:
    body = client.get("/styles/top", params={"season": "SS2020", "limit": 10}).json()
    assert body["season"] == "SS2020" and body["observed"] and body["cutoff"] == "2020-05-27"
    first = body["styles"][0]
    assert first["style_id"] == "0599580" and first["actual_units"] == 14674 and first["actual_rank"] == 1
    # the AW2020 default list has no actuals
    assert client.get("/styles/top", params={"limit": 1}).json()["styles"][0]["actual_units"] is None


def test_ss2020_detail_has_actual_and_no_concept(client: TestClient) -> None:
    d = client.get("/styles/751471", params={"season": "ss2020"}).json()  # season id is case-insensitive
    assert d["season"] == "SS2020" and d["concept"] is None
    assert d["actual"]["units"] is not None and len(d["actual"]["weekly"]) == 4
    assert d["explanation"]["source"].startswith("models/lgbm_final_20200527")


def test_unknown_season_is_404(client: TestClient) -> None:
    for url in ("/styles/top?season=AW2099", "/styles/0751471?season=AW2099"):
        r = client.get(url)
        assert r.status_code == 404 and r.json()["season"] == "AW2099" and "Available" in r.json()["detail"]


def test_model_summary(client: TestClient) -> None:
    m = client.get("/model/summary").json()
    assert m["success_definition"] and m["horizon"] and m["stock_caveat"]
    reg = {r["method"]: r for r in m["regressor"]["backtest"]}
    assert reg["LightGBM regressor"]["ndcg@50"]["mean"] == 0.924
    clf = {r["method"]: r for r in m["classifier_top0.1pct"]["methods"]}
    assert clf["LightGBM classifier"]["validation"]["pr_auc"] == 0.456
    assert m["classifier_top1pct"]["reliability_plot_url"] == "/images/classifier/reliability.png"
    assert client.get(m["classifier_top1pct"]["reliability_plot_url"]).status_code == 200
    assert m["seasonal"]["ss2020_top10_hits"] == 7
