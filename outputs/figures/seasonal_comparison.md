# Seasonal comparison: what the model expects to win

Same pipeline (retrain on all fully-observed cutoffs, predict next 4 weeks) at two cutoffs. Category mix = share of predicted units within each run's top-100 styles.

#### summer 2020 — cutoff 2020-05-27 (predicting 2020-05-27 → 2020-06-23)

| # | style | name | type | garment group | pred units | last 4w | actual | actual rank |
|---|---|---|---|---|---|---|---|---|
| 1 | 0599580 | Timeless Midrise Brief | Swimwear bottom | Swimwear | 12,703 | 11,050 | 14,674 | 1 |
| 2 | 0854677 | C Lolly Top | Bikini top | Swimwear | 8,901 | 5,687 | 1,811 | 101 |
| 3 | 0854683 | C Lolly Midrise Cheeky Brief | Swimwear bottom | Swimwear | 8,681 | 6,127 | 2,972 | 42 |
| 4 | 0684209 | Simple as That Triangle Top | Bikini top | Swimwear | 8,591 | 7,757 | 7,293 | 3 |
| 5 | 0610776 | Tilly (1) | T-shirt | Jersey Basic | 8,570 | 7,544 | 7,495 | 2 |
| 6 | 0688537 | Simple as that Cheeky Tanga | Swimwear bottom | Swimwear | 7,290 | 6,729 | 6,347 | 7 |
| 7 | 0554598 | Nora T-shirt | T-shirt | Jersey Basic | 6,327 | 5,208 | 6,286 | 8 |
| 8 | 0759871 | Tilda tank | Vest top | Jersey Basic | 6,225 | 5,441 | 5,847 | 10 |
| 9 | 0854678 | C Lolly Bandeau | Bikini top | Swimwear | 6,089 | 4,234 | 1,936 | 91 |
| 10 | 0776237 | Shake it in Balconette | Bikini top | Swimwear | 5,514 | 4,845 | 6,972 | 5 |

Backtest: 7/10 of the predicted top-10 were in the actual top-10.

#### autumn 2020 — cutoff 2020-09-23 (predicting 2020-09-23 → 2020-10-20)

| # | style | name | type | garment group | pred units | last 4w |
|---|---|---|---|---|---|---|
| 1 | 0751471 | Pluto RW slacks (1) | Trousers | Trousers | 7,417 | 9,182 |
| 2 | 0706016 | Jade HW Skinny Denim TRS | Trousers | Trousers | 6,854 | 8,198 |
| 3 | 0762846 | Lucy blouse | Shirt | Blouses | 6,492 | 6,662 |
| 4 | 0685814 | RICHIE HOOD | Hoodie | Jersey Basic | 5,486 | 6,108 |
| 5 | 0456163 | Woody hoodie | Hoodie | Jersey Basic | 4,768 | 4,616 |
| 6 | 0573085 | Madison skinny HW | Trousers | Trousers | 4,633 | 3,306 |
| 7 | 0685813 | PETAR SWEATSHIRT | Sweater | Jersey Basic | 4,496 | 4,545 |
| 8 | 0884319 | Lucien CONSCIOUS | Blouse | Blouses | 4,398 | 4,997 |
| 9 | 0915529 | Liliana | Sweater | Knitwear | 4,045 | 4,570 |
| 10 | 0673677 | Henry polo. (1) | Sweater | Knitwear | 3,917 | 3,202 |

#### Category mix of the top-100 (product group, share of predicted units)

| product group | summer 2020 | autumn 2020 | change (pts) |
|---|---|---|---|
| Garment Upper body | 31% | 63% | +32.5 |
| Garment Lower body | 21% | 30% | +9.7 |
| Socks & Tights | 2% | 2% | +0.4 |
| Underwear | 3% | 1% | -1.7 |
| Unknown | 0% | 1% | +0.9 |
| Nightwear | 0% | 1% | +0.8 |
| Garment Full body | 6% | 1% | -5.1 |
| Accessories | 0% | 1% | +0.6 |
| Swimwear | 38% | 0% | -38.1 |

**What changed:** from summer 2020 to autumn 2020, Garment Upper body gains the most share of the predicted top-100 (+32.5 pts) and Swimwear loses the most (-38.1 pts).
