# Merchmix: next-season styles

Forecast next-season winning styles from H&M transactions, then generate new product concepts with an agent-based workflow.

![Final board: 3 forecast winners and their new concepts](outputs/final_board.png)

**Evidence per winner:** sales curve, top SHAP drivers, reference → concept with critic verdict, KEEP/CHANGE,
and the backtest summary ([`outputs/evidence_sheet.png`](outputs/evidence_sheet.png)).

![Evidence sheet: from forecast to concept for each winner](outputs/evidence_sheet.png)

## Results

| # | Style | Forecast units, 23 Sep–20 Oct 2020 | What changed (visible in the concept) | Critic |
|---|---|---|---|---|
| 1 | Pluto RW slacks (trousers, `0751471`) | 7,417 | Charcoal-and-camel glen check; camel side stripe | approved |
| 2 | Lucy blouse (shirt, `0762846`) | 6,492 | Burgundy satin; fuller sleeves gathered into deep cuffs | approved |
| 3 | RICHIE HOOD (hoodie, `0685814`) | 5,486 | Heather-grey fabric; twin zip pockets; contrast-tipped cuffs | not approved |

The hoodie's intended cropped, boxy shape was not produced by the image model, so the critic did not approve it.
Full lineage per style: [`outputs/evidence/<code>/lineage.json`](outputs/evidence/).

## Model

- **Style** = `product_code` (all colourways of one design). **Target** = units sold in the next 4 weeks.
- **LightGBM** trained on 45 weekly snapshots, predicting the uplift over the naive "last week × 4" run-rate.
- **Backtest** (10 rolling weekly cutoffs, NDCG@50): model 0.924 vs 0.906 for both last-week × 4 and last-4-weeks,
  7/10 cutoff wins against each. Last-4-weeks is beaten clearly only on the validation week (0.862 vs 0.569).
  Where the model disagrees with last-week × 4, its demotions of fading styles were right 6/10, its promotions 3/10.

Details: [`outputs/figures/eval_table.md`](outputs/figures/eval_table.md), [`movers.md`](outputs/figures/movers.md), [WRITEUP.md](WRITEUP.md).

## Architecture

```mermaid
flowchart LR
  O{{Orchestrator<br/>Claude Agent SDK}} --> FC[forecaster] & AN[style analyst] & DE[designer] & CR[critic]
  SK[/style-dna-brief skill/] --> AN
  FC --> F["forecast MCP<br/>(LightGBM + SHAP)"]
  FC --> R["retail MCP<br/>(DuckDB data)"]
  AN --> R
  DE --> I["image MCP"]
  CR --> I
  I --> GEN["FLUX.1 Kontext<br/>image edit"]
  I --> CLIP["CLIP<br/>novelty check"]
```

- **Forecaster**: picks the top-3 (one per garment group, sold in the last 2 weeks), explains them with SHAP, writes `forecast.json`.
- **Style analyst**: uses the `style-dna-brief` skill to write a KEEP/CHANGE brief and image prompt; a validator checks it.
- **Designer**: edits the best-selling reference photo with FLUX.1 Kontext (reuses existing images to save GPU quota).
- **Critic**: CLIP similarity to the style's own photos plus a visual check; approves or gives one revision note.
- **Guard-rails in code**: tool whitelist per agent, writes only under `outputs/`, at most 4 new images and 1 revision
  per run; every tool call is logged to [`outputs/runs/20260928-005623/trace.jsonl`](outputs/runs/20260928-005623/trace.jsonl).

## Seasonal bonus

Same pipeline at a late-May cutoff: 7/10 of the predicted summer top-10 were in the actual top-10. From summer to
autumn, swimwear falls from 38% to 0% of the predicted top-100 and upper-body garments rise from 31% to 63%.
See [`outputs/figures/seasonal_comparison.md`](outputs/figures/seasonal_comparison.md).

## How to run (Windows PowerShell, Python 3.11)

```powershell
# 1. Setup
py -3.11 -m venv .venv; .venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # add HF_TOKEN for real images (free Hugging Face account)

# 2. Data (needs a Kaggle API token and accepting the competition rules)
$c = "h-and-m-personalized-fashion-recommendations"
.venv\Scripts\kaggle competitions download -c $c -f transactions_train.csv -p raw
.venv\Scripts\kaggle competitions download -c $c -f articles.csv -p raw
.venv\Scripts\python scripts\convert_data.py --raw raw --out data

# 3. Forecast: training snapshots, model + evaluation, top-3 + evidence + reference photos
.venv\Scripts\python -m forecasting.features
.venv\Scripts\python -m forecasting.model
.venv\Scripts\python -m forecasting.select

# 4. Agent run: --mock uses a local stand-in instead of the image model (no image API key)
.venv\Scripts\python -m agents.orchestrator --cutoff 2020-09-22 --mock
.venv\Scripts\python -m agents.orchestrator --cutoff 2020-09-22

# 5. Finalize: lineage, review sheet, board (needs outputs/runs/<run_id>/captions.json), evidence sheet, tests
.venv\Scripts\python scripts\finalize_run.py --run <run_id> --final 0751471=concept_2.png 0762846=concept_1.png 0685814=concept_2.png
.venv\Scripts\python scripts\evidence_sheet.py --run <run_id>
.venv\Scripts\python -m pytest
```

The agents (step 4, including `--mock`) need Claude: `ANTHROPIC_API_KEY` in `.env` or a local Claude Code login.
Reference photos (`outputs/refs/`) are H&M/Kaggle data and are not included; step 3 downloads them.

## Repository layout

```
forecasting/        data (DuckDB), features, LightGBM model, evaluation, top-3 selection
mcp_servers/        FastMCP servers: retail, forecast, image
agents/             orchestrator, in-process bookkeeping tools, sub-agent prompts
image/              image generation (HF Space / Replicate / mock) with quota guards, CLIP, board layout
skills_lib/         style-dna-brief template and validator (works without the SDK)
.claude/skills/     the style-dna-brief skill
scripts/            data conversion, MCP smoke test, image test, finalize, evidence sheet
tests/              leakage, NDCG, quota guards, skill examples (21 tests)
notebooks/          EDA script
outputs/            board, evidence sheet, per-style evidence, figures, models, agent run
```

## Limitations

- **No stock data.** "Sold in the last 2 weeks" is only a proxy for availability; sold-out styles look like weak sellers.
- **4-week horizon.** It covers the opening of autumn, not a full season; longer horizons were not validated.
- **Hoodie shape not achieved.** The image model changed fabric, colour and trims but not the garment's proportions.
- **CLIP is weak on colour.** Colourways of the same style score 0.80–0.94, so the critic also checks images visually.

More in [WRITEUP.md](WRITEUP.md): approach, why 4 weeks, full results, lineage, limitations, issues hit.
Evidence: [`outputs/evidence/`](outputs/evidence/) · agent run: [`outputs/runs/20260928-005623/`](outputs/runs/20260928-005623/).
