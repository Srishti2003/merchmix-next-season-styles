# Winner classifier (top 0.1% label): evaluation

Success = a style's units over the next 4 weeks rank in the **top 0.1% of styles active at that cutoff** (threshold computed per cutoff; ties at the threshold count as winners). Model: LightGBM binary on the regressor's 23 features, 135 boosting rounds (early stopping on the last 4 training cutoffs, metric average precision).

## Winner base rate

- Mean base rate over the 11 scored cutoffs: **0.1027%** (20 winners out of 19,306 active styles on average; range 19–21 winners).
- Units needed to be a winner (next 4 weeks): 2,544–4,754 depending on the cutoff (median 3,290).

## Rolling backtest: 10 weekly cutoffs (2020-05-27 → 2020-07-29), mean ± std

Same folds as the regressor's backtest: at cutoff c every model (classifier and regressor) is retrained only on cutoffs whose 4-week target ended by c. Every method is ranked by its own score; the classifier by raw probability.

| method | pr_auc | precision@20 | precision@50 | recall@50 |
|---|---|---|---|---|
| LightGBM classifier | 0.807 ± 0.052 | 0.735 ± 0.063 | 0.358 ± 0.018 | 0.910 ± 0.062 |
| Last week × 4 | 0.750 ± 0.084 | 0.680 ± 0.075 | 0.354 ± 0.027 | 0.899 ± 0.073 |
| Last 4 weeks units | 0.743 ± 0.072 | 0.675 ± 0.075 | 0.352 ± 0.019 | 0.894 ± 0.058 |
| LightGBM regressor (units) | 0.796 ± 0.094 | 0.735 ± 0.071 | 0.360 ± 0.019 | 0.915 ± 0.060 |

- classifier wins vs Last week × 4: pr_auc 6/10 (ties 0); precision@20 5/10 (ties 3); precision@50 2/10 (ties 6); recall@50 2/10 (ties 6)
- classifier wins vs Last 4 weeks units: pr_auc 8/10 (ties 0); precision@20 7/10 (ties 3); precision@50 3/10 (ties 6); recall@50 3/10 (ties 6)
- classifier wins vs LightGBM regressor (units): pr_auc 4/10 (ties 0); precision@20 4/10 (ties 3); precision@50 1/10 (ties 7); recall@50 1/10 (ties 7)

## Validation cutoff 2020-08-26 (trained on 2019-09-25 → 2020-07-29)

| method | pr_auc | precision@20 | precision@50 | recall@50 |
|---|---|---|---|---|
| LightGBM classifier | 0.456 | 0.450 | 0.320 | 0.762 |
| Last week × 4 | 0.566 | 0.600 | 0.380 | 0.905 |
| Last 4 weeks units | 0.229 | 0.300 | 0.220 | 0.524 |
| LightGBM regressor (units) | 0.514 | 0.550 | 0.380 | 0.905 |

## Does the classifier beat last week × 4? (backtest means)

- pr_auc: classifier 0.807 vs last week × 4 0.750 (higher; classifier ahead at 6/10 cutoffs)
- precision@20: classifier 0.735 vs last week × 4 0.680 (higher; classifier ahead at 5/10 cutoffs)
- precision@50: classifier 0.358 vs last week × 4 0.354 (higher; classifier ahead at 2/10 cutoffs)
- recall@50: classifier 0.910 vs last week × 4 0.899 (higher; classifier ahead at 2/10 cutoffs)

## Calibration (out-of-time)

The isotonic map used at each cutoff is fitted only on out-of-fold predictions whose labels were known at that cutoff (OOF cutoff ≤ cutoff − 4 weeks). 8 extra OOF cutoffs before the backtest window are used only as calibration data. Brier and ECE per cutoff, mean over the cutoffs:

| probabilities | Brier (backtest) | ECE (backtest) | Brier (validation) | ECE (validation) |
|---|---|---|---|---|
| raw | 0.00043 | 0.00041 | 0.00087 | 0.00081 |
| isotonic-calibrated | 0.00042 | 0.00044 | 0.00085 | 0.00081 |

Calibrated beats raw on Brier at 8/11 cutoffs and on ECE at 6/11.

Pooled reliability over all scored cutoffs (reliability_top0.1pct.png):

| probabilities | bin | n | mean predicted | observed |
|---|---|---|---|---|
| raw | 0.00–0.01 | 211,551 | 0.0000 | 0.0001 |
| raw | 0.01–0.03 | 209 | 0.0179 | 0.0287 |
| raw | 0.03–0.10 | 221 | 0.0566 | 0.0452 |
| raw | 0.10–0.20 | 89 | 0.1396 | 0.1348 |
| raw | 0.20–0.40 | 76 | 0.2891 | 0.2632 |
| raw | 0.40–0.60 | 39 | 0.5037 | 0.2308 |
| raw | 0.60–0.80 | 35 | 0.6897 | 0.4000 |
| raw | 0.80–1.00 | 144 | 0.9307 | 0.8681 |
| calibrated | 0.00–0.01 | 210,942 | 0.0001 | 0.0000 |
| calibrated | 0.01–0.03 | 806 | 0.0149 | 0.0285 |
| calibrated | 0.03–0.10 | 237 | 0.0493 | 0.0338 |
| calibrated | 0.10–0.20 | 111 | 0.1284 | 0.1532 |
| calibrated | 0.20–0.40 | 93 | 0.2499 | 0.2796 |
| calibrated | 0.40–0.60 | 6 | 0.4682 | 0.6667 |
| calibrated | 0.60–0.80 | 44 | 0.6847 | 0.5000 |
| calibrated | 0.80–1.00 | 125 | 0.9496 | 0.8800 |

## Final scoring: cutoff 2020-09-23 (forecast window 2020-09-23 → 2020-10-20)

Classifier retrained on all cutoffs 2019-09-25 → 2020-08-26; calibrator fitted on 371,999 OOF rows (cutoffs up to 2020-08-26, targets ending by 2020-09-22). Among the regressor's top-20 styles, 3 have a calibrated probability ≥ 0.999 and there are 10 distinct calibrated values (range 0.038–1.000). The final ranking and prediction_score are set in final_scores.md.

Metric definitions: pr_auc = average precision (tie-aware); precision@k = share of winners among the k highest scores; recall@50 = share of all winners found in the top 50; Brier = mean squared error of the probability; ECE = n-weighted mean |predicted − observed| over the bins in evaluate.CAL_BINS.
