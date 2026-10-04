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
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.model_service import ModelService, StyleNotFound
from backend.schemas import ErrorResponse, Health, StyleDetail, TopStylesResponse, ValidationErrorResponse

IMAGE_DIRS = {"refs", "evidence"}                                       # sub-folders of outputs/ served as images
IMAGE_FILES = {"generated_concepts.png", "final_board.png", "evidence_sheet.png"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

app = FastAPI(
    title="Merchmix style intelligence API",
    version="1.0.0",
    description="Predicted winning styles for 23 Sep – 20 Oct 2020 (H&M data): ranked list, per-style explanation, "
                "sales history and the generated next-season concepts. Serves `outputs/predictions.json`, written "
                "offline by `python -m data_science.predict`.",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_methods=["GET"], allow_headers=["*"])

NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown or malformed style_id"}}
INVALID = {422: {"model": ValidationErrorResponse, "description": "Query parameter out of range"}}


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    """One ModelService per process: predictions.json is read once."""
    return ModelService()


@app.exception_handler(StyleNotFound)
async def style_not_found(_: Request, exc: StyleNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.detail, "style_id": exc.style_id})


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": "Invalid request parameters.",
                                                                   "errors": errors}))


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


@app.get("/health", response_model=Health, summary="Service status and model version", tags=["meta"])
def health(service: ModelService = Depends(get_service)) -> dict:
    """Status, model version, prediction cutoff date and the number of styles available."""
    return service.health()


@app.get("/styles/top", response_model=TopStylesResponse, responses=INVALID, tags=["styles"],
         summary="Top predicted styles (ranked list)")
def top_styles(limit: int = Query(10, ge=1, le=200, description="number of styles to return (1-200)"),
               offset: int = Query(0, ge=0, description="number of styles to skip"),
               service: ModelService = Depends(get_service)) -> dict:
    """The selected top 3 first (one per garment group), then every other style by forecast units.
    Each item has the scores, category, the last 8 weeks of sales plus the 4-week total, and a photo URL
    (null when the catalogue photo has not been downloaded)."""
    return service.top(limit, offset)


@app.get("/styles/{style_id}", response_model=StyleDetail, response_model_by_alias=True, responses=NOT_FOUND,
         tags=["styles"], summary="One style: product info, scores, explanation, history, concept")
def style_detail(style_id: str = PathParam(description="product_code, with or without the leading zero "
                                                       "(751471 or 0751471)"),
                 service: ModelService = Depends(get_service)) -> dict:
    """Product information and attributes, scores, the 5 SHAP drivers in plain words plus a one-line reason for
    (non-)selection, up to 26 weeks of sales, and for the top 3 the generated concept (image, KEEP/CHANGE brief,
    critic status)."""
    return service.detail(style_id)


@app.get("/images/{path:path}", tags=["images"], summary="Static images from outputs/",
         responses={404: {"model": ErrorResponse, "description": "Image not found or not served"}})
def image(path: str, service: ModelService = Depends(get_service)) -> FileResponse:
    """Reference photos (`refs/…`), concepts and sales curves (`evidence/…`) and the final boards. Only image files
    under those locations are served."""
    root = service.outputs_dir.resolve()
    target = (root / path).resolve()
    parts = Path(path).parts
    allowed = bool(parts) and ((parts[0] in IMAGE_DIRS and len(parts) > 1) or (len(parts) == 1 and path in IMAGE_FILES))
    if not (allowed and target.is_relative_to(root) and target.suffix.lower() in IMAGE_TYPES and target.is_file()):
        raise StarletteHTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(target)
