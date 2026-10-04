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
seed, model; input photos in `outputs/refs/` are H&M/Kaggle data, downloaded by `data_science/select.py` and not included) → `critic.jsonl` (decision, CLIP scores, note) → `lineage.json` (links all of these, plus the
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

## 9. Full-stack phase 1: success definition, winner classifier, extended EDA
Numbers in this section come from `outputs/classifier/{eval_classifier.md, eval_classifier_top0.1pct.md,
final_scores.md}` and `data_science/notebooks/02_eda_extended.ipynb`.

**Success definition.** A style is a *winner* if its units in the next 4 weeks rank in the **top 1% of styles
active at that cutoff** (sold in the previous 12 weeks). The threshold is set per cutoff, not globally, so it moves
with the season (857–1,512 units across the scored cutoffs; about 194 winners out of ~19,300 active styles). Top 1%
matches the decision: a buyer can follow up on a short list, not thousands of styles. And because it's a rank
rather than a fixed unit count, it isn't distorted by seasonal swings or by the 2020 COVID dip.

**Why 4 weeks.** See §2. In short, styles turn over fast (median 19 active weeks, 32% of units from styles under 12
weeks old). Four weeks is the longest window in which "which existing styles will lead" can be tested honestly, and
from the 23 Sep cutoff those 4 weeks are the opening of autumn.

**Stock limitation.** There's no stock or availability data, so observed sales are *censored demand*: a style that
sold out, or was never fully ranged in a store, looks like a weak seller. The model learns these as decliners and
will under-rank styles that were held back by supply. The reverse also happens: a style with deep stock and a
markdown can look like a riser. "Sold in the last 2 weeks" removes only styles that are clearly gone. No inventory
assumptions were added.

**Extra data that would help, in order of value:** stock on hand and availability by store/size (to un-censor
demand); the markdown calendar and planned promotions (to separate price-driven from organic demand); returns (net
units, and to catch fit problems); web traffic such as views, add-to-cart and searches (an early demand signal before
sales); store footfall (to normalise store sales); size curves (to see whether a style is limited by broken sizes).

**Classifier result.** A LightGBM binary classifier on the same 23 features, with the same 10-cutoff rolling backtest
and no leakage. The isotonic calibrator used at each cutoff is fitted only on out-of-fold predictions whose labels
were already known there.
- **It beats the naive baselines:** backtest PR-AUC 0.770 vs 0.738 for last week × 4 (ahead at 9/10 cutoffs) and
  0.713 for last 4 weeks (10/10). Precision@50 is 0.986 vs 0.938.
- **It ties the regressor:** PR-AUC 0.770 vs 0.771, with the classifier ahead at 5/10 cutoffs. Precision@50 gives
  5 wins and 4 ties.
- **Its value is a calibrated probability, not a better ranking.** Isotonic calibration lowers ECE from 0.00361 to
  0.00282 (backtest) and from 0.00271 to 0.00225 (validation). Raw probabilities above 0.4 were overconfident.
- **Saturation:** at the final cutoff, P(top 1%) is ≥ 0.999 for 17 of the regressor's top-20 styles, so it can't
  separate them.
- **Stricter top-0.1% label (about 20 winners per cutoff).** Its calibrated probabilities spread out where P(top 1%)
  saturates: 10 distinct values among the regressor's top 20, from 0.038 to 1.000. Calibration changed its backtest
  ECE from 0.00041 to 0.00044, which is within noise (2 standard errors of the per-cutoff change = 0.00006), and
  improved Brier at 8 of 11 cutoffs. **This is the displayed `prediction_score`.**
- **Its weakness:** on the validation week the top-0.1% classifier ranks worse than last week × 4 (PR-AUC 0.456 vs
  0.566) and the regressor (0.514). In the backtest it is ahead of last week × 4 on PR-AUC at only 6 of 10
  cutoffs. So it is a **relative-strength signal**: how likely the style is to be among the very top ~20, given
  everything the model knows. It is not a better ranker.
- **What the scores mean in the outputs:** the ranking and the top 3 stay the regressor's forecast units.
  `prediction_score` is the calibrated P(top 0.1%): Pluto 1.000, Lucy 0.942, RICHIE 1.000. `confidence_top1pct` is
  the calibrated P(top 1%), 1.000 for all three.

**Why perennial basics such as Jade HW Skinny Denim and Cat Tee are not in the top 3.**
- *The rule allows one style per garment group.* Jade is a trouser, and Pluto, also a trouser, has the higher
  forecast (7,417 vs 6,854 units).
- *Forecast volume.* Cat Tee is in the same garment group as RICHIE (Jersey Basic) and has a much lower forecast
  (2,682 units, regressor rank 29, vs 5,486). It reached the classifier's top 3 only because P(top 1%) ties at 1.000
  for 24 styles, and tiny differences in raw probability decided the order.
- *Design value.* Jade has sold in 99 consecutive weeks and sold 12,527 units in the same 4 weeks last year. A
  never-out-of-stock basic tells the design team little that's new. This is a judgement, not part of the rule:
  RICHIE has also sold for 97 weeks.

**EDA findings (extended notebook).**
1. **Duplicate rows are multi-unit purchases, not errors.** 9.36% of transaction rows are exact repeats of another
   row; the raw CSV with the real customer_id gives the same count. They occur on every day (6.1–15.7% of daily
   rows), and their frequency falls steeply with group size, with small bumps at 6 and 8. Each row is kept as one
   unit.
2. **Customers are bimodal by age:** peaks at 21 and 51, 1.16% missing. Ages 25–34 buy 36.4% of units.
3. **Online share is 66.7–74.8% of units in every age band**, highest for 25–34. Trousers, dresses and sweaters
   lead the categories in every band.
4. **Data quality:** no missing dates (734 of 734 days); 995 articles (390 styles) never sold; 0.39% of articles
   have no description; `product_code` matches `article_id // 1000` for every article, so `style_id` is stable.
   Prices are scaled (max ≈ 0.59), with 0.26% low and 0.04% high outliers within product type, which are kept.

**Sampling.** Full transactions, articles and customers. Images only for the 3 winners (9 photos; the full set is
about 30 GB). The forecast does not use images, so its ranking is unaffected. The concepts see only each winner's
three best-selling colours.
