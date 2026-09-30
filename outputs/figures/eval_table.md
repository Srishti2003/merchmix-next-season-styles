# Evaluation: LightGBM vs naive baselines

Primary model: **LightGBM regressor** — chosen on the inner early-stopping cutoffs (NDCG@50 regressor 0.944 vs ranker 0.846), not on the validation week, to avoid selection bias.

### Validation cutoff 2020-08-26 (target = units 2020-08-26 → 2020-09-22, 20,639 styles)

| method | precision@3 | precision@12 | top-12 hit rate in top-50 | ndcg@50 | spearman_top500 | wape_top500 |
|---|---|---|---|---|---|---|
| LightGBM regressor | **0.667** | 0.417 | **1.000** | 0.862 | **0.520** | 0.473 |
| LightGBM ranker | **0.667** | 0.417 | 0.583 | 0.752 | 0.427 | n/a |
| Last 4 weeks units | 0.000 | 0.250 | 0.500 | 0.569 | 0.387 | 0.612 |
| Same 4 weeks last year | **0.667** | 0.167 | 0.583 | 0.601 | 0.216 | 0.777 |
| Last week × 4 | **0.667** | **0.500** | **1.000** | **0.864** | 0.506 | **0.446** |

### Rolling backtest: 10 weekly cutoffs (2020-05-27 → 2020-07-29), mean ± std

At each cutoff the regressor is retrained only on cutoffs whose 4-week target ended before it (94 boosting rounds, fixed).

| method | ndcg@50 | precision@12 |
|---|---|---|
| LightGBM regressor | 0.924 ± 0.031 | 0.725 ± 0.125 |
| Last week × 4 | 0.906 ± 0.040 | 0.708 ± 0.119 |
| Last 4 weeks units | 0.906 ± 0.016 | 0.658 ± 0.092 |

- vs Last week × 4 — ndcg@50: model wins **7/10** (ties 0); precision@12: model wins **3/10** (ties 6)
- vs Last 4 weeks units — ndcg@50: model wins **7/10** (ties 0); precision@12: model wins **7/10** (ties 2)

The regressor predicts log-uplift over the naive 'last week × 4' run-rate, weighted toward high-volume styles (a plain log-units regressor lost to that baseline at the top of the list). Where the model disagrees with that baseline and who was right: see movers.md.

precision@k = share of predicted top-k in actual top-k; top-12 hit rate in top-50 = share of the actual top-12 that appear in the predicted top-50; WAPE is n/a for the ranker (no unit forecast).
