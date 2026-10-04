# Winner classifier (top 1% label): evaluation

Success = a style's units over the next 4 weeks rank in the **top 1% of styles active at that cutoff** (threshold computed per cutoff; ties at the threshold count as winners). Model: LightGBM binary on the regressor's 23 features, 294 boosting rounds (early stopping on the last 4 training cutoffs, metric average precision).

## Winner base rate

- Mean base rate over the 11 scored cutoffs: **1.0035%** (194 winners out of 19,306 active styles on average; range 186–207 winners).
- Units needed to be a winner (next 4 weeks): 857–1,512 depending on the cutoff (median 1,047).

## Rolling backtest: 10 weekly cutoffs (2020-05-27 → 2020-07-29), mean ± std

Same folds as the regressor's backtest: at cutoff c every model (classifier and regressor) is retrained only on cutoffs whose 4-week target ended by c. Every method is ranked by its own score; the classifier by raw probability.

| method | pr_auc | precision@20 | precision@50 | recall@50 |
|---|---|---|---|---|
| LightGBM classifier | 0.770 ± 0.036 | 1.000 ± 0.000 | 0.986 ± 0.016 | 0.256 ± 0.007 |
| Last week × 4 | 0.738 ± 0.047 | 0.990 ± 0.021 | 0.938 ± 0.047 | 0.244 ± 0.013 |
| Last 4 weeks units | 0.713 ± 0.033 | 0.980 ± 0.035 | 0.954 ± 0.021 | 0.248 ± 0.006 |
| LightGBM regressor (units) | 0.771 ± 0.041 | 0.995 ± 0.016 | 0.976 ± 0.018 | 0.254 ± 0.007 |

- classifier wins vs Last week × 4: pr_auc 9/10 (ties 0); precision@20 2/10 (ties 8); precision@50 8/10 (ties 1); recall@50 8/10 (ties 1)
- classifier wins vs Last 4 weeks units: pr_auc 10/10 (ties 0); precision@20 3/10 (ties 7); precision@50 10/10 (ties 0); recall@50 10/10 (ties 0)
- classifier wins vs LightGBM regressor (units): pr_auc 5/10 (ties 0); precision@20 1/10 (ties 9); precision@50 5/10 (ties 4); recall@50 5/10 (ties 4)

## Validation cutoff 2020-08-26 (trained on 2019-09-25 → 2020-07-29)

| method | pr_auc | precision@20 | precision@50 | recall@50 |
|---|---|---|---|---|
| LightGBM classifier | 0.702 | 1.000 | 0.960 | 0.232 |
| Last week × 4 | 0.687 | 0.950 | 0.980 | 0.237 |
| Last 4 weeks units | 0.402 | 0.900 | 0.720 | 0.174 |
| LightGBM regressor (units) | 0.675 | 1.000 | 0.960 | 0.232 |

## Does the classifier beat last week × 4? (backtest means)

- pr_auc: classifier 0.770 vs last week × 4 0.738 (higher; classifier ahead at 9/10 cutoffs)
- precision@20: classifier 1.000 vs last week × 4 0.990 (higher; classifier ahead at 2/10 cutoffs)
- precision@50: classifier 0.986 vs last week × 4 0.938 (higher; classifier ahead at 8/10 cutoffs)
- recall@50: classifier 0.256 vs last week × 4 0.244 (higher; classifier ahead at 8/10 cutoffs)

## Calibration (out-of-time)

The isotonic map used at each cutoff is fitted only on out-of-fold predictions whose labels were known at that cutoff (OOF cutoff ≤ cutoff − 4 weeks). 8 extra OOF cutoffs before the backtest window are used only as calibration data. Brier and ECE per cutoff, mean over the cutoffs:

| probabilities | Brier (backtest) | ECE (backtest) | Brier (validation) | ECE (validation) |
|---|---|---|---|---|
| raw | 0.00495 | 0.00361 | 0.00506 | 0.00271 |
| isotonic-calibrated | 0.00469 | 0.00282 | 0.00517 | 0.00225 |

Calibrated beats raw on Brier at 6/11 cutoffs and on ECE at 7/11.

Pooled reliability over all scored cutoffs (reliability.png):

| probabilities | bin | n | mean predicted | observed |
|---|---|---|---|---|
| raw | 0.00–0.01 | 203,977 | 0.0002 | 0.0005 |
| raw | 0.01–0.03 | 2,642 | 0.0176 | 0.0367 |
| raw | 0.03–0.10 | 2,022 | 0.0555 | 0.0890 |
| raw | 0.10–0.20 | 775 | 0.1450 | 0.1716 |
| raw | 0.20–0.40 | 794 | 0.2866 | 0.2720 |
| raw | 0.40–0.60 | 516 | 0.4931 | 0.3391 |
| raw | 0.60–0.80 | 440 | 0.7007 | 0.4636 |
| raw | 0.80–1.00 | 1,198 | 0.9438 | 0.8464 |
| calibrated | 0.00–0.01 | 200,027 | 0.0003 | 0.0003 |
| calibrated | 0.01–0.03 | 5,379 | 0.0164 | 0.0204 |
| calibrated | 0.03–0.10 | 2,749 | 0.0527 | 0.0535 |
| calibrated | 0.10–0.20 | 1,556 | 0.1458 | 0.1832 |
| calibrated | 0.20–0.40 | 983 | 0.2764 | 0.3001 |
| calibrated | 0.40–0.60 | 560 | 0.4971 | 0.4982 |
| calibrated | 0.60–0.80 | 393 | 0.6586 | 0.7226 |
| calibrated | 0.80–1.00 | 717 | 0.9351 | 0.9331 |

## Final scoring: cutoff 2020-09-23 (forecast window 2020-09-23 → 2020-10-20)

Classifier retrained on all cutoffs 2019-09-25 → 2020-08-26; calibrator fitted on 371,999 OOF rows (cutoffs up to 2020-08-26, targets ending by 2020-09-22). Among the regressor's top-20 styles, 17 have a calibrated probability ≥ 0.999 and there are 4 distinct calibrated values (range 0.893–1.000). The final ranking and prediction_score are set in final_scores.md.

Metric definitions: pr_auc = average precision (tie-aware); precision@k = share of winners among the k highest scores; recall@50 = share of all winners found in the top 50; Brier = mean squared error of the probability; ECE = n-weighted mean |predicted − observed| over the bins in evaluate.CAL_BINS.
