# Write-up: next-season style forecast → AI product concepts (H&M data)

**Result:** `outputs/final_board.png` (3 winners → 3 concepts) · evidence: `outputs/evidence_sheet.png`,
`outputs/evidence/<code>/lineage.json` · agent trace: `outputs/runs/20260928-005623/trace.jsonl`.

## 1. Approach in one paragraph
Transactions (31.8M rows, Sep 2018 – 22 Sep 2020) are aggregated to **styles** (`product_code`, i.e. all colour
variants of one design) per Wed→Tue week. A LightGBM model, trained on 45 weekly snapshots, forecasts each
style's units for the next 4 weeks. The top-3 are picked with a diversity rule (one per garment group) and an
availability proxy (must have sold in the last 2 weeks). An agentic workflow then turns each winner into a new
product concept. Claude Agent SDK orchestrator → forecaster, style-analyst (reusable `style-dna-brief`
skill), designer (FLUX.1 Kontext image edit) and critic (CLIP novelty + visual check) sub-agents, all working
through MCP tools. Every step is written to `lineage.json`.

## 2. Key choices
- **Style = `product_code`.** Colourways of one design share the same "DNA"; ranking articles would give three
  colours of one trouser. Winners sell in 10–23 colours, which is also the evidence that *shape*, not colour,
  is what customers buy.
- **"Strong" = most units in the next 4 weeks** (buyers are tracked too). Units is what a merchandiser plans.
- **Why a 4-week horizon, and how that relates to "next season".**
  - *Fast fashion turns over quickly.* The median style sells for 19 active weeks, and 32% of weekly units
    come from styles launched in the last 12 weeks (EDA). Over a full 12–13-week season, a large share of
    demand comes from styles that don't exist yet, so a style-level history model can't forecast them. Four
    weeks is the longest window where "which existing styles will lead" is still well defined and testable.
  - *The cutoff is the season boundary.* Data ends 22 Sep 2020, so the 4 weeks (23 Sep – 20 Oct) are the
    opening of autumn. The seasonal bonus shows the same pipeline switches with the season: at a late-May
    cutoff swimwear is 38% of the top-100's predicted units, and in late September it's 0%, with upper-body
    garments up 32 points.
  - *"Next season" is carried by the concepts, not the forecast.* The forecast finds the styles winning as
    the season opens. The generated concepts are next-season *products* that keep those styles' proven DNA,
    i.e. input to the next buying cycle, not a forecast of it.
  - *Honest limit.* The model is not validated beyond 4 weeks. `HORIZON_WEEKS` is one config value, but a
    12-week run was not tested, so no accuracy claim is made for it.
- **Weekly snapshots, no leakage.** Features use only weeks before the cutoff (tests scramble future data and
  check the features don't change). The target is the 4 weeks from the cutoff.
- **Model target = uplift over the naive run-rate.** A plain log-units regressor lost to "last week × 4" at the
  top of the ranking, so the model predicts `log1p(next 4w) − log1p(4 × last week)`, weighted toward
  high-volume styles. A LambdaRank variant was tried and lost (NDCG@50 0.846 vs 0.944 on the inner cutoffs).
- **Selection:** at most one style per garment group (a board of three trousers is not useful). "Sold in the
  last 2 weeks" is the *only* availability signal. **The dataset has no stock data** (see §6).

## 3. Model results: honest version
Validation week (cutoff 26 Aug → 22 Sep 2020, 20,639 styles) and a 10-cutoff rolling backtest (each cutoff
scored by a model trained only on earlier data):

| | Model | Last week × 4 | Last 4 weeks | Same 4 wks last year |
|---|---|---|---|---|
| NDCG@50, validation week | 0.862 | **0.864** | 0.569 | 0.601 |
| precision@12, validation week | 0.417 | **0.500** | 0.250 | 0.167 |
| NDCG@50, backtest mean ± std | **0.924 ± 0.031** | 0.906 ± 0.040 | 0.906 ± 0.016 | n/a |
| Model wins on NDCG@50 (backtest) | n/a | **7/10** | **7/10** | n/a |

- Against **"last 4 weeks"** (which lags momentum) the win is clear on the validation week (0.862 vs 0.569),
  but in the 10-week backtest it's the same small margin as against last-week × 4 (0.924 vs 0.906, 7/10 wins).
  Against the strong naive baseline **"last week × 4" it wins 7/10 backtest weeks on NDCG@50 but only ties it
  on the validation week**, and on precision@12 it wins 3/10 (6 ties).
- **Where the value is (`outputs/figures/movers.md`):** when the model ranks a style far *below* the
  baseline it is usually right (6/10, mostly one-week spikes that fade); when it ranks one far *above* it is
  usually wrong (3/10, mostly late-summer swimwear). So the model's value is **demoting fading styles, not
  finding new risers.**
- SHAP (`shap_summary.png`): one-week spikes get shrunk; heavily discounted, end-of-life styles are faded;
  young styles get a boost. All three winners share the same top upward driver, *still near full price*.
- **The three winners share a pattern** (checked against forecast.json and the weekly sales): all are
  early-autumn risers (last 4 weeks = 1.9–2.6× the 4 weeks before) whose September rise the model partly
  discounts (forecast −19%, −2%, −10% vs the last 4 weeks). They win on sustained volume at near full price
  (1–2% below peak price), not markdowns.
- Seasonal bonus: at the summer cutoff, 7 of the predicted top-10 were in the actual top-10.

## 4. From forecast to concept (lineage)
For each winner, `outputs/evidence/<code>/` holds `forecast.json` (rank, forecast, SHAP drivers, attributes) →
`brief.json` (KEEP/CHANGE, validated by `skills_lib/style_dna.py`) → `generation_log.jsonl` (exact prompt,
seed, model; input photos in `outputs/refs/` are H&M/Kaggle data, downloaded by `forecasting/select.py` and not included) → `critic.jsonl` (decision, CLIP scores, note) → `lineage.json` (links all of these, plus the
board caption and the briefed changes the image did not render).

| Winner | Forecast (next 4 wks) | Final concept | Critic | CLIP to own photos |
|---|---|---|---|---|
| #1 Pluto RW slacks (trousers) | 7,417 | concept_2: glen check, camel side stripe | approved | 0.726 |
| #2 Lucy blouse | 6,492 | concept_1: burgundy satin, fuller gathered sleeves | approved | 0.690 |
| #3 RICHIE HOOD | 5,486 | concept_2: grey, zip pockets, striped cuffs | **not approved** | 0.797 |

**Manual interventions (approved by the user, logged with notes):** (1) a third RICHIE HOOD attempt beyond the
one-revision rule, also rejected, so the board keeps concept_2 and says the critic did not approve it;
(2) Pluto regenerated from the run's own brief so its lineage is clean (the run had reused an image made from a
hand-written test prompt). Board captions were checked by eye; briefed changes the image model did not render
(e.g. Pluto's cropped hem, RICHIE's cropped boxy shape) are not claimed.

## 5. Agentic workflow
Orchestrator (Claude Agent SDK, Sonnet) with 4 sub-agents, 3 stdio MCP servers (`retail`, `forecast`,
`image`) plus an in-process `run` server for deterministic bookkeeping, and 1 skill. Guard-rails live in code,
not prompts: a permission gate (whitelist; writes only under `outputs/`), an image budget (≤ 4 new images and
≤ 1 revision per run; identical requests reuse the file), and lineage assembled from files, not typed by the
LLM. The run: 7 min, 59 tool calls across 5 agents, 3 GPU generations.

## 6. Limitations
- **No stock data → censored demand.** A style that sold out looks like a weak seller; "sold in the last 2
  weeks" only filters out styles that are clearly gone. Real use needs inventory/OTB data.
- **New styles have no history**, and 32% of units come from them. This model can only rank existing styles.
- **2020 is COVID-distorted** (online share 69% → 73%); last-year features are weaker than usual.
- **Image model:** FLUX Kontext changes fabric, colour and trims well but **did not change a garment's
  proportions** in three hoodie attempts, and small chest labels reappear.
- **CLIP is a weak novelty proxy:** it barely reacts to colour (colourways of one style score 0.80–0.94), so the
  "too close" threshold was calibrated to 0.80 and the critic also checks the images visually.
- **Brief validator gap:** Pluto's brief KEEPs "solid colour" while CHANGING to a check. The rule check does not
  catch KEEP-vs-CHANGE contradictions yet.
- Single retailer, single season boundary; the backtest covers 10 summer weeks.

## 7. Issues we hit (engineering)
- **scikit-learn replaced because of a device policy.** The laptop's Application Control policy blocks
  sklearn's compiled DLL (`sklearn.utils.murmurhash`). Its only use, `ndcg_score`, is now a pure-numpy
  function with identical behaviour (tie-averaged gains, log2 discount, 0 when the ideal DCG is 0, mean over
  rows). It reproduces the sklearn-computed validation NDCG@50 values exactly, and tests pin hard-coded
  values, **so all reported results are unchanged.** (`shap` still imports sklearn, so only the two SHAP *plots*
  can't be regenerated on that laptop; the SHAP drivers use LightGBM's built-in TreeSHAP.)
- Reproducibility: DuckDB's `mode()` tie-breaking and multithreaded LightGBM made reruns drift; both are fixed.
- Agent SDK: sub-agents run asynchronously in this CLI version, and an unanswerable permission prompt blocked
  the critic in a dry run. Fixed with a code-based permission gate and a harness that waits for all sub-agents.
  Four mock dry runs were done before spending real GPU quota.
- Free Hugging Face GPU quota (~5–10 images/day) instead of a paid API: hence the budget guards.

## 8. What I'd do next at Merchmix
Plug in inventory and open-to-buy so selection is stock-aware; add new-style cold-start (attribute-similarity
to past launches); route concepts to a buyer approval queue with the lineage attached; and extend the critic
with a shape check (silhouette/edge comparison) instead of relying on CLIP.
