# Merchmix Next-Season Concept Run — Summary

**Run ID:** 20260928-005623
**Cutoff:** 2020-09-22 (forecast window: 23 Sep – 20 Oct 2020)
**Board composed:** No (COMPOSE_BOARD=no)

> Note on availability: the dataset has no stock/availability data. "Sold in the last 2 weeks" is used by the
> forecaster as the only proxy for whether a style is still available to sell.

---

## Winner #1 — Pluto RW slacks (1) — product_code 0751471
- **Type / group:** Trousers / Trousers
- **Forecast:** 7,417.3 predicted units (next 4 weeks) vs. naive run-rate 6,844 units; 9,182 units sold in the last 4 weeks.
- **Top SHAP drivers:** last-week units (1,711) pulls forecast down ×0.62 vs. run-rate; momentum (last wk vs 4wk avg = 0.75) pulls down ×0.74; discount vs. highest price seen (2%) raises ×1.21 vs. run-rate.
- **Brief KEEP:** slim tapered ankle-length cigarette leg; regular waist with belt loops, zip fly, concealed elastication; front slant pockets, fake back pockets, pressed centre crease; smooth stretch-weave tailored fabric in a solid colour.
- **Brief CHANGE:** solid black → brushed wool-look houndstooth check in charcoal and camel; plain side seam → narrow camel contrast side stripe with exposed metal-tip belt loops; closed ankle hem → cropped hem with a small side split.
- **Final concept:** `outputs/evidence/0751471/concept_1.png` (an existing concept was reused — no GPU cost — per the designer's no-duplicate-generation rule).
- **Critic decision:** **Approve**. Novelty check verdict `ok`, max CLIP similarity to refs = 0.6801. Note: "Keeps the tapered trouser silhouette, waistband/belt-loop and slanted-pocket construction while clearly reading as a new product via the charcoal/camel check, contrast side stripe, and cropped hem with visible side split."
- **Caveat:** the reused concept's original generation prompt was simpler than this run's brief (missing "houndstooth" wording and "fake back pockets"/"metal-tip belt loops" specifics); `write_lineage` flagged `prompt_matches_brief: false`. The critic judged the image itself as satisfying the brief visually despite the prompt-text mismatch.
- **Revision:** none needed.

## Winner #2 — Lucy blouse — product_code 0762846
- **Type / group:** Shirt / Blouses
- **Forecast:** 6,492.3 predicted units (next 4 weeks) vs. naive run-rate 6,820 units; 6,662 units sold in the last 4 weeks.
- **Top SHAP drivers:** last-week units (1,705) pulls forecast down ×0.62 vs. run-rate; momentum (last wk vs 4wk avg = 1.02) pulls down ×0.64; discount vs. highest price seen (1%) raises ×1.23 vs. run-rate.
- **Brief KEEP:** long-sleeved woven blouse with shirt collar and open V-neck; button-front placket and buttoned cuffs; relaxed straight body with rounded hem; solid single-colour base.
- **Brief CHANGE:** straight sleeve → voluminous balloon sleeves gathered into deep three-button cuffs; matte woven fabric → fluid satin with soft sheen; black → deep burgundy.
- **Final concept:** `outputs/evidence/0762846/concept_1.png` (newly generated — no prior concept existed).
- **Critic decision:** **Approve**. Novelty check verdict `ok`, max CLIP similarity to refs = 0.6903. Note: "Same collar/V-neck placket/button-front/rounded-hem silhouette preserved, with clearly new deep burgundy colourway, glossy satin sheen, and visibly fuller gathered sleeve reflecting the brief's CHANGE items." (The specific "three-button cuff" detail wasn't distinctly resolvable at image scale, but multiple other CHANGE items were clearly visible.)
- **Revision:** none needed. `write_lineage` confirmed `prompt_matches_brief: true`.

## Winner #3 — RICHIE HOOD — product_code 0685814
- **Type / group:** Hoodie / Jersey Basic
- **Forecast:** 5,485.6 predicted units (next 4 weeks) vs. naive run-rate 4,660 units; 6,108 units sold in the last 4 weeks.
- **Top SHAP drivers:** last-week units (1,165) pulls forecast down ×0.62 vs. run-rate; momentum (last wk vs 4wk avg = 0.76) pulls down ×0.75; discount vs. highest price seen (1%) raises ×1.24 vs. run-rate.
- **Brief KEEP:** relaxed fit pullover hoodie silhouette; lined drawstring hood; kangaroo pocket with ribbing at cuffs/hem; brushed cotton-blend sweatshirt fabric.
- **Brief CHANGE:** solid black → brushed melange heather grey with subtle tonal fleck texture; flat kangaroo pocket/plain cuffs → twin zip-entry pockets with contrast-tipped ribbed cuffs; regular hip-length/set-in sleeves → cropped boxy length with dropped oversized shoulder seams.
- **Concept attempt 1:** `outputs/evidence/0685814/concept_1.png` (newly generated). Critic: **revise** — novelty check verdict `too_close` (max CLIP similarity 0.8002, at the too-close threshold). Note: colourway/fleck/pockets/cuffs read well, but body/sleeve silhouette barely differentiated from the reference.
- **Revision (the one allowed per this run):** designer regenerated with an explicit instruction to dramatically crop the hem and exaggerate dropped/batwing shoulders (seed 43) → `outputs/evidence/0685814/concept_2.png`.
- **Concept attempt 2 (final judgment):** Critic decision: **revise** (recorded honestly, but per run rules no further revision is executed — this is final). Novelty check verdict improved to `ok`, max CLIP similarity = 0.7969 (down from 0.8002, just under the too_close line). Note: "CLIP clears the too_close threshold by a hair (0.797 vs 0.80), but the hem still reads as standard hip-length and the shoulders/sleeves taper normally rather than showing the requested dramatic crop or batwing drop; DNA (melange fleck, drawstring hood, twin zip pockets, contrast cuffs) is preserved, but the silhouette ask wasn't fully realized."
- **Final concept selected:** `outputs/evidence/0685814/concept_2.png` — chosen over concept_1 as the better of the two per critic/novelty scores (verdict moved from `too_close` to `ok`; no concept was ever fully approved for this style). `write_lineage` recorded `critic_decision: revise`, `prompt_matches_brief: false`.
- **Outcome flag:** this style's final concept did NOT reach an "approve" verdict within the run's one-revision budget. It ships as the best available candidate, with the silhouette differentiation weaker than the brief intended.

---

## Revision budget used
- 1 of 1 allowed revisions used this run, on RICHIE HOOD (0685814).
- Image generation budget: 2 new images generated this run (Lucy blouse concept_1, RICHIE HOOD concept_2) + 1 reused (Pluto RW slacks concept_1) + 1 prior existing (RICHIE HOOD concept_1, generated before this run's designer step) — within the 4-new-image / 1-revision run budget.

## Tool errors
- None encountered. No GPU-quota refusals; no validate_brief resubmissions needed (all three briefs validated on first attempt).

## Data caveat
- No stock/availability data exists in this dataset. "Sold in the last 2 weeks" is the forecaster's only proxy for a style still being available/sellable, per the tool's own documentation.
