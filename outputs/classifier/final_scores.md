# Final scores at cutoff 2020-09-23 (forecast window 2020-09-23 → 2020-10-20)

Ranking = the regressor's forecast units (published rule: one style per garment group, sold in the last 2 weeks). The classifiers only supply probabilities.

## Is calibrated P(top 0.1%) usable as prediction_score?

- Among the regressor's top 20: 3 styles at ≥ 0.999, 10 distinct values (range 0.038–1.000) → spread out.
- Backtest ECE raw 0.00041 vs calibrated 0.00044.
- Fewest winners at any scored cutoff: 19.
- Decision: prediction_score = **calibrated P(top 1%)** (ties broken by forecast units); confidence_top1pct = calibrated P(top 1%).

## Top 10 by regressor forecast units

| reg rank | style | name | garment group | forecast units | P(top 0.1%) calibrated | P(top 1%) calibrated | top-3 |
|---|---|---|---|---|---|---|---|
| 1 | 0751471 | Pluto RW slacks (1) | Trousers | 7,417 | 1.000 | 1.000 | #1 |
| 2 | 0706016 | Jade HW Skinny Denim TRS | Trousers | 6,854 | 1.000 | 1.000 |  |
| 3 | 0762846 | Lucy blouse | Blouses | 6,492 | 0.942 | 1.000 | #2 |
| 4 | 0685814 | RICHIE HOOD | Jersey Basic | 5,486 | 1.000 | 1.000 | #3 |
| 5 | 0456163 | Woody hoodie | Jersey Basic | 4,768 | 0.760 | 1.000 |  |
| 6 | 0573085 | Madison skinny HW | Trousers | 4,633 | 0.876 | 1.000 |  |
| 7 | 0685813 | PETAR SWEATSHIRT | Jersey Basic | 4,496 | 0.727 | 1.000 |  |
| 8 | 0884319 | Lucien CONSCIOUS | Blouses | 4,398 | 0.254 | 1.000 |  |
| 9 | 0915529 | Liliana | Knitwear | 4,045 | 0.038 | 0.976 |  |
| 10 | 0673677 | Henry polo. (1) | Knitwear | 3,917 | 0.760 | 1.000 |  |

Top-3: #1 0751471 Pluto RW slacks (1) (prediction_score 1.000, 7,417 units), #2 0762846 Lucy blouse (prediction_score 1.000, 6,492 units), #3 0685814 RICHIE HOOD (prediction_score 1.000, 5,486 units)
