"""HTTP layer only: routing, validation, error JSON, CORS and static images. Logic lives in model_service.

  uvicorn backend.api:app --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Path as PathParam, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.model_service import DEFAULT_SEASON, SeasonNotFound, SeasonRegistry, StyleNotFound
from backend.schemas import (ErrorResponse, Health, ModelSummary, SeasonsResponse, StyleDetail, TopStylesResponse,
                             ValidationErrorResponse)

IMAGE_DIRS = {"refs", "evidence", "classifier"}                         # sub-folders of outputs/ served as images
IMAGE_FILES = {"generated_concepts.png", "final_board.png", "evidence_sheet.png"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

app = FastAPI(
    title="Merchmix style intelligence API",
    version="1.0.0",
    description="Predicted winning styles (H&M data): ranked list, per-style explanation, sales history and the "
                "generated next-season concepts. AW2020 (default) is the forecast for 23 Sep – 20 Oct 2020; SS2020 "
                "is a backtest season (27 May – 23 Jun 2020) with actual units. Serves the files written offline by "
                "`python -m data_science.predict` and `python -m data_science.summary`.",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_methods=["GET"], allow_headers=["*"])

NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown or malformed style_id, or unknown season"}}
SEASON_NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown season"}}
INVALID = {422: {"model": ValidationErrorResponse, "description": "Query parameter out of range"}}


@lru_cache(maxsize=1)
def get_registry() -> SeasonRegistry:
    """One registry per process: each season's predictions file is read once."""
    return SeasonRegistry()


def season_param(season: str = Query(DEFAULT_SEASON, description="season id from /seasons (AW2020 or SS2020)")) -> str:
    return season


@app.exception_handler(StyleNotFound)
async def style_not_found(_: Request, exc: StyleNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.detail, "style_id": exc.style_id})


@app.exception_handler(SeasonNotFound)
async def season_not_found(_: Request, exc: SeasonNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.detail, "season": exc.season})


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": "Invalid request parameters.",
                                                                   "errors": errors}))


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.get("/health", response_model=Health, summary="Service status and model version", tags=["meta"])
def health(registry: SeasonRegistry = Depends(get_registry)) -> dict:
    """Status, model version, prediction cutoff date and the number of styles available (default season)."""
    return registry.health()


@app.get("/seasons", response_model=SeasonsResponse, summary="Available seasons", tags=["meta"])
def seasons(registry: SeasonRegistry = Depends(get_registry)) -> dict:
    """AW2020 = the forecast (default); SS2020 = backtest at the 27 May 2020 cutoff, with actual units."""
    return {"seasons": registry.seasons()}


@app.get("/model/summary", response_model=ModelSummary, response_model_by_alias=True, tags=["meta"],
         summary="Model performance vs baselines, success definition, horizon, stock caveat")
def model_summary(registry: SeasonRegistry = Depends(get_registry)) -> dict:
    """Regressor and classifiers vs the naive baselines (from the published evaluation files), calibration,
    the reliability plot URL and the SS2020 → AW2020 category-mix shift."""
    return registry.summary()


@app.get("/styles/top", response_model=TopStylesResponse, responses={**INVALID, **SEASON_NOT_FOUND}, tags=["styles"],
         summary="Top predicted styles (ranked list)")
def top_styles(limit: int = Query(10, ge=1, le=200, description="number of styles to return (1-200)"),
               offset: int = Query(0, ge=0, description="number of styles to skip"),
               season: str = Depends(season_param),
               registry: SeasonRegistry = Depends(get_registry)) -> dict:
    """The selected top 3 first (one per garment group), then every other style by forecast units.
    Each item has the scores, category, the last 8 weeks of sales plus the 4-week total, and a photo URL
    (null when the catalogue photo has not been downloaded). Observed seasons add actual units and rank."""
    return registry.service(season).top(limit, offset)


@app.get("/styles/{style_id}", response_model=StyleDetail, response_model_by_alias=True, responses=NOT_FOUND,
         tags=["styles"], summary="One style: product info, scores, explanation, history, concept")
def style_detail(style_id: str = PathParam(description="product_code, with or without the leading zero "
                                                       "(751471 or 0751471)"),
                 season: str = Depends(season_param),
                 registry: SeasonRegistry = Depends(get_registry)) -> dict:
    """Product information and attributes, scores, the 5 SHAP drivers in plain words plus a one-line reason for
    (non-)selection, up to 26 weeks of sales, and for the top 3 the generated concept (image, KEEP/CHANGE brief,
    critic status). Observed seasons add the actual units, rank and weekly values in the window."""
    return registry.service(season).detail(style_id)


@app.get("/images/board-ref/{style_id}.png", tags=["images"], response_class=Response,
         summary="A top-3 winner's reference photo, cropped from the committed concept board",
         responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}})
def board_reference(style_id: str, registry: SeasonRegistry = Depends(get_registry)) -> Response:
    """Fallback product photo when outputs/refs/ (git-ignored Kaggle photos) is absent, e.g. on the hosted demo.
    Only the 3 winners shown on the committed board have one."""
    return Response(registry.service().board_ref_png(style_id), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/images/{path:path}", tags=["images"], summary="Static images from outputs/",
         responses={404: {"model": ErrorResponse, "description": "Image not found or not served"}})
def image(path: str, registry: SeasonRegistry = Depends(get_registry)) -> FileResponse:
    """Reference photos (`refs/…`), concepts and sales curves (`evidence/…`), reliability plots (`classifier/…`)
    and the final boards. Only image files under those locations are served."""
    root = registry.outputs_dir.resolve()
    target = (root / path).resolve()
    parts = Path(path).parts
    allowed = bool(parts) and ((parts[0] in IMAGE_DIRS and len(parts) > 1) or (len(parts) == 1 and path in IMAGE_FILES))
    if not (allowed and target.is_relative_to(root) and target.suffix.lower() in IMAGE_TYPES and target.is_file()):
        raise StarletteHTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(target)
