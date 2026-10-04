# API

FastAPI service over the offline predictions in `outputs/predictions.json` (written by `python -m data_science.predict`).
Prediction cutoff 2020-09-23; forecast window 2020-09-23 → 2020-10-20. The 200 styles with the highest forecast
units are available; the selected top 3 come first.

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Interactive docs with schemas and example responses: `http://localhost:8000/docs` (OpenAPI: `/openapi.json`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, model version, prediction cutoff, number of styles |
| GET | `/styles/top?limit=10&offset=0` | ranked list (top 3 first, then by forecast units) |
| GET | `/styles/{style_id}` | one style: product info, scores, explanation, 26-week history, concept (top 3) |
| GET | `/images/{path}` | images from `outputs/`: `refs/…`, `evidence/…`, `generated_concepts.png` |

Code layout: `backend/api.py` (HTTP only), `backend/model_service.py` (loads the predictions once, no FastAPI
imports), `backend/schemas.py` (Pydantic response models). CORS allows `http://localhost:8501` (Streamlit).

## Scores

| field | meaning |
|---|---|
| `rank` | 1–3 for the selected winners (highest forecast units, one per garment group, sold in the last 2 weeks); `null` otherwise |
| `forecast_rank` | position by forecast units among all 20,318 scored styles |
| `forecast_units` | regressor forecast of units sold in the forecast window |
| `prediction_score` | calibrated probability that the style is a top-0.1% seller (≈ top 20 styles) over the next 4 weeks. A relative-strength signal; the ranking is `forecast_units` |
| `confidence_top1pct` | calibrated probability of being a top-1% seller over the next 4 weeks |

## GET /health

```bash
curl localhost:8000/health
```
```json
{"status": "ok", "model_version": "regressor-20200923-r94+clf-top1pct-r294+clf-top0.1pct-r135",
 "prediction_cutoff": "2020-09-23", "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
 "n_styles": 200, "n_styles_scored": 20318}
```

## GET /styles/top

Query parameters: `limit` (1–200, default 10), `offset` (≥ 0, default 0).

```bash
curl "localhost:8000/styles/top?limit=2"
```
```json
{
  "cutoff": "2020-09-23",
  "forecast_window": {"start": "2020-09-23", "end": "2020-10-20"},
  "total": 200, "limit": 2, "offset": 0,
  "styles": [
    {
      "style_id": "0751471", "name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
      "prediction_score": 1.0, "confidence_top1pct": 1.0, "forecast_units": 7417.3,
      "category": {"product_type": "Trousers", "garment_group": "Trousers"},
      "sales_history": {
        "last_8_weeks": [{"week_start": "2020-07-29", "units": 764}, {"week_start": "2020-08-05", "units": 616},
                         {"week_start": "2020-08-12", "units": 1291}, {"week_start": "2020-08-19", "units": 2070},
                         {"week_start": "2020-08-26", "units": 2796}, {"week_start": "2020-09-02", "units": 2659},
                         {"week_start": "2020-09-09", "units": 2016}, {"week_start": "2020-09-16", "units": 1711}],
        "units_last_4w": 9182
      },
      "image_url": "/images/refs/0751471/0751471001.jpg"
    },
    {"style_id": "0762846", "name": "Lucy blouse", "rank": 2, "forecast_rank": 3, "prediction_score": 0.9419, "...": "..."}
  ]
}
```

`image_url` is the catalogue photo of the best-selling colourway, or `null` when it has not been downloaded
(reference photos are Kaggle data and are not in the repository); the frontend shows a placeholder then.

Errors: `limit=0`, `limit=201`, `offset=-1` or a non-integer → **422**:
```json
{"detail": "Invalid request parameters.",
 "errors": [{"loc": ["query", "limit"], "msg": "Input should be less than or equal to 200", "type": "less_than_equal"}]}
```

## GET /styles/{style_id}

`style_id` is the 7-digit `product_code`; the leading zero is optional (`751471` = `0751471`).

```bash
curl localhost:8000/styles/751471
```
```json
{
  "style_id": "0751471", "name": "Pluto RW slacks (1)", "rank": 1, "forecast_rank": 1,
  "prediction_score": 1.0, "confidence_top1pct": 1.0, "p_top0_1pct": 1.0, "forecast_units": 7417.3,
  "category": {"product_type": "Trousers", "garment_group": "Trousers"},
  "attributes": {"product_group_name": "Garment Lower body", "garment_group_name": "Trousers",
                 "index_group_name": "Ladieswear", "section_name": "Womens Everyday Collection",
                 "colour_group_name": "Black", "graphical_appearance_name": "Solid",
                 "detail_desc": "Ankle-length cigarette trousers in a stretch weave with a zip fly, …", "n_colours": 10},
  "performance": {"units_last_1w": 1711, "units_last_2w": 3727, "units_last_4w": 9182, "units_last_12w": 17012,
                  "units_same_4w_last_year": 3159, "units_to_date": 64305, "buyers_last_4w": 6367,
                  "weeks_since_launch": 67, "discount_vs_peak_price": 0.0192, "online_share_last_4w": 0.6788},
  "explanation": {
    "why_selected": "Selected #1: highest forecast in Trousers (7,417 units for 2020-09-23 to 2020-10-20), sold in the last 2 weeks; P(top 0.1%) = 1.00.",
    "reasons": [
      {"text": "Units sold last week is 1,711: this lowers the forecast by 38% compared with repeating last week's sales for 4 weeks.",
       "feature": "units sold last week", "value": "1,711", "direction": "lowers", "effect_pct": -38.0,
       "raw": "units sold last week = 1,711 → lowers the forecast ×0.62 vs the last-week run-rate"},
      "… 4 more"
    ],
    "source": "outputs/evidence/0751471/forecast.json"
  },
  "sales_history_26w": [{"week_start": "2020-03-25", "units": 1659, "buyers": 1121}, "…",
                        {"week_start": "2020-09-16", "units": 1711, "buyers": 1178}],
  "image_url": "/images/refs/0751471/0751471001.jpg",
  "reference_image_urls": ["/images/refs/0751471/0751471001.jpg", "/images/refs/0751471/0751471042.jpg",
                           "/images/refs/0751471/0751471041.jpg"],
  "concept": {
    "image_url": "/images/evidence/0751471/concept_2.png",
    "reference_image_url": "/images/refs/0751471/0751471001.jpg",
    "keep": [{"trait": "slim tapered ankle-length cigarette leg", "evidence": "detail_desc '…'; sold in 10 colourways …"}, "…"],
    "change": [{"axis": "pattern", "from": "solid black", "to": "brushed wool-look houndstooth check in charcoal and camel"}, "…"],
    "what_changed": "Charcoal-and-camel glen-check wool-look fabric; camel contrast side stripe down each leg. …",
    "critic": {"decision": "approve", "status": "Critic: approved", "note": "Same slim tapered ankle-length trouser silhouette …",
               "max_similarity_to_references": 0.7263,
               "changes_not_visible": ["cropped hem with a small side split (still reads as ankle length)", "exposed metal-tip belt loops"]}
  }
}
```

- The 5 `reasons` are the regressor's TreeSHAP drivers; each is a multiplier on the naive "last week × 4" forecast.
- `why_selected` also explains non-selection, e.g. `GET /styles/0706016` (Jade, forecast rank 2): *"Forecast rank #2
  (6,854 units); not in the top 3 because Trousers is already represented by #1 Pluto RW slacks (1) (7,417 units
  forecast)."*
- `concept` is `null` for every style outside the top 3. For RICHIE (`0685814`) the critic did not approve the final
  concept (`"decision": "revise"`, `"status": "Critic: not approved"`): the image model did not change the hoodie's
  proportions.
- `sales_history_26w` starts at the first sale for styles launched less than 26 weeks before the cutoff.

Errors → **404** with the requested id:
```json
{"detail": "Style 0000001 is not in the published predictions (top 200 styles by forecast units).", "style_id": "0000001"}
{"detail": "Malformed style_id 'abc': expected up to 7 digits (a product_code, e.g. 0751471).", "style_id": "abc"}
```

## GET /images/{path}

Serves image files (`.jpg`, `.png`) from `outputs/refs/`, `outputs/evidence/` and the boards
(`generated_concepts.png`, `final_board.png`, `evidence_sheet.png`). Use the URLs returned by the other endpoints.

```bash
curl -o concept.png localhost:8000/images/evidence/0751471/concept_2.png      # 200 image/png
curl localhost:8000/images/cache/train.parquet                                 # 404
```
```json
{"detail": "Image not found: cache/train.parquet"}
```

Anything else (other folders, non-image files, missing files, paths escaping `outputs/`) → **404** with the JSON above.

## Error format

| status | when | body |
|---|---|---|
| 404 | unknown or malformed `style_id` | `{"detail": "...", "style_id": "..."}` |
| 404 | image not found / not served, unknown route | `{"detail": "..."}` |
| 422 | query parameter out of range or wrong type | `{"detail": "Invalid request parameters.", "errors": [{"loc", "msg", "type"}]}` |

## Reference photos for the demo

Photos are not committed (H&M/Kaggle data). Winners: `python -m data_science.select` (or the `download_refs` one-liner
in the README). For the list view, `python scripts/fetch_list_photos.py --n 50` fetches one photo per style for the top 50 into
`outputs/refs/<style_id>/` with the same per-image Kaggle download (best-selling colourway first, next colourway
when Kaggle has no image for it). Missing photos give `image_url: null`.
