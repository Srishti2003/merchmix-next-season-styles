"""Build and execute notebooks/02_eda_extended.ipynb (customer behaviour + data quality), outputs saved.

  python scripts/build_eda_extended.py

Reads the Parquet files in HM_DATA_DIR and customers.csv from HM_RAW_DIR (default: the parent of HM_DATA_DIR).
pandas/duckdb/matplotlib only.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "02_eda_extended.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# 02 — Extended EDA: customer behaviour and data quality

Complements `01_eda.py` (seasonality, category mix, style lifetime). Everything here is computed from the full
transactions (31.8M rows), articles (105,542) and customers files with DuckDB and pandas.
Run order: `scripts/convert_data.py` first; `customers.csv` is read from the raw-data folder."""),
    ("code", """import os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

import config
from forecasting import viz

viz.apply_style()
pd.set_option("display.width", 140, "display.max_columns", 20)
RAW = Path(os.getenv("HM_RAW_DIR") or config.DATA_DIR.parent)
con = duckdb.connect(config={"memory_limit": "3GB"})  # progress bar is off by default in Jupyter
TX = f"read_parquet('{config.TX_GLOB}')"
ART = f"read_parquet('{config.ARTICLES_PATH}')"
CUST = f"read_csv_auto('{(RAW / 'customers.csv').as_posix()}', types={{'customer_id': 'VARCHAR'}})"
# transactions store cust = CAST(hash(customer_id) AS UBIGINT) (scripts/convert_data.py); same hash here
con.execute(f"CREATE TEMP TABLE cust AS SELECT *, CAST(hash(customer_id) AS UBIGINT) AS cust FROM {CUST}")
con.execute(f\"\"\"CREATE TEMP TABLE c_tx AS SELECT cust, count(*) AS n_tx, count(DISTINCT t_dat) AS n_days,
               avg(CAST(channel = 2 AS DOUBLE)) AS online_share FROM {TX} GROUP BY cust\"\"\")
q = lambda sql: con.execute(sql).df()
q("SELECT (SELECT count(*) FROM cust) AS customers, (SELECT count(*) FROM c_tx) AS buying_customers, "
  "(SELECT count(*) FROM c_tx WHERE cust NOT IN (SELECT cust FROM cust)) AS buyers_not_in_customers_csv")"""),

    ("md", "## 1. Customer behaviour\n### Age: distribution and missing share"),
    ("code", """age = q("SELECT age FROM cust")["age"]
print(f"customers: {len(age):,} | age missing: {age.isna().sum():,} ({age.isna().mean():.2%})")
print(age.describe().round(1).to_string())
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.hist(age.dropna(), bins=range(16, 100), color=viz.SERIES[0])
ax.set_xlabel("age"); ax.set_ylabel("customers")
ax.set_title("Customer age (customers.csv)"); plt.show()
by_age = age.value_counts().sort_index()
print(f"main peak: age {by_age.idxmax()} ({by_age.max():,}); trough 35-45 at age {by_age.loc[35:45].idxmin()} "
      f"({by_age.loc[35:45].min():,}); second peak 45-60: age {by_age.loc[45:60].idxmax()} ({by_age.loc[45:60].max():,})")"""),
    ("md", "The age distribution is bimodal: a large peak in the early twenties, a dip in the late thirties and a second, "
           "lower peak around 50 (exact ages printed above)."),

    ("md", "### club_member_status and fashion_news_frequency"),
    ("code", """for col in ("club_member_status", "fashion_news_frequency"):
    vc = q(f"SELECT coalesce({col}, '(missing)') AS value, count(*) AS n FROM cust GROUP BY 1 ORDER BY n DESC")
    vc["share"] = (vc["n"] / vc["n"].sum()).map("{:.2%}".format)
    print(col); print(vc.to_string(index=False)); print()"""),
    ("md", "`fashion_news_frequency` has both `NONE` and `None` spellings, plus missing values; treat all three as 'no newsletter'."),

    ("md", "### Repeat purchase and basket size\nRepeat purchase = bought on at least 2 different days in the 2 years. "
           "Basket = all rows by one customer on one day (the data has no order id)."),
    ("code", """rep = q(\"\"\"SELECT count(*) AS buyers, avg(CAST(n_days >= 2 AS DOUBLE)) AS repeat_rate,
              median(n_days) AS median_days, avg(n_days) AS mean_days, median(n_tx) AS median_rows FROM c_tx\"\"\")
never = q("SELECT avg(CAST(cust NOT IN (SELECT cust FROM c_tx) AS DOUBLE)) AS share FROM cust")["share"][0]
print(rep.round(3).to_string(index=False))
print(f"customers in customers.csv with no purchase at all: {never:.2%}")
basket = q(f"SELECT n, count(*) AS baskets FROM (SELECT cust, t_dat, count(*) AS n FROM {TX} GROUP BY 1, 2) GROUP BY n ORDER BY n")
w = basket["baskets"]
mean = (basket["n"] * w).sum() / w.sum()
cum = w.cumsum() / w.sum()
print(f"customer-days: {w.sum():,} | mean basket {mean:.2f} rows | median {basket['n'][cum >= 0.5].iloc[0]} | "
      f"p90 {basket['n'][cum >= 0.9].iloc[0]} | single-item baskets {w[basket['n'] == 1].sum() / w.sum():.1%}")"""),

    ("md", "### Online vs store share and top categories by age band\nChannel 2 = online, 1 = store (convention used in data.py)."),
    ("code", """BAND = \"\"\"CASE WHEN c.age IS NULL THEN 'missing' WHEN c.age < 25 THEN '16-24' WHEN c.age < 35 THEN '25-34'
          WHEN c.age < 45 THEN '35-44' WHEN c.age < 55 THEN '45-54' ELSE '55+' END\"\"\"
con.execute(f"CREATE TEMP TABLE band AS SELECT cust, {BAND} AS age_band FROM cust c")
ch = q(f\"\"\"SELECT b.age_band, count(*) AS units, avg(CAST(t.channel = 2 AS DOUBLE)) AS online_share,
           count(DISTINCT t.cust) AS buyers
           FROM {TX} t JOIN band b USING (cust) GROUP BY 1 ORDER BY 1\"\"\")
ch["units_share"] = ch["units"] / ch["units"].sum()
print(ch.assign(online_share=ch.online_share.map("{:.1%}".format), units_share=ch.units_share.map("{:.1%}".format)).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(ch["age_band"], ch["online_share"], color=viz.SERIES[0])
ax.set_ylim(0, 1); ax.set_ylabel("online share of units"); ax.set_title("Online share by age band"); plt.show()"""),
    ("code", """cat = q(f\"\"\"SELECT b.age_band, a.product_type_name, count(*) AS units
            FROM {TX} t JOIN band b USING (cust) JOIN {ART} a USING (article_id) GROUP BY 1, 2\"\"\")
cat["share"] = cat["units"] / cat.groupby("age_band")["units"].transform("sum")
top = (cat.sort_values(["age_band", "share"], ascending=[True, False]).groupby("age_band").head(5)
       .assign(item=lambda d: d.product_type_name + " " + d.share.map("{:.1%}".format)))
for band, s in top.groupby("age_band")["item"]:
    print(f"{band:8} " + " | ".join(s))
idx = q(f\"\"\"SELECT b.age_band, a.index_group_name, count(*) AS units
            FROM {TX} t JOIN band b USING (cust) JOIN {ART} a USING (article_id) GROUP BY 1, 2\"\"\")
print(); print(idx.pivot(index="age_band", columns="index_group_name", values="units")
              .pipe(lambda p: p.div(p.sum(axis=1), axis=0)).map("{:.1%}".format).to_string())"""),

    ("md", "## 2. Data quality\n### Exact duplicate transaction rows"),
    ("code", """dup = q(f\"\"\"SELECT n, count(*) AS groups FROM (SELECT t_dat, cust, article_id, price, channel, count(*) AS n
            FROM {TX} GROUP BY ALL) WHERE n > 1 GROUP BY n ORDER BY n\"\"\")
total = q(f"SELECT count(*) AS n FROM {TX}")["n"][0]
rows_in_dups = int((dup["n"] * dup["groups"]).sum())
extra = int(((dup["n"] - 1) * dup["groups"]).sum())
print(f"rows: {total:,} | rows that belong to a duplicate group: {rows_in_dups:,} ({rows_in_dups / total:.2%}) | "
      f"rows beyond the first in each group: {extra:,} ({extra / total:.2%})")
print(dup.head(8).to_string(index=False))
daily = q(f\"\"\"WITH g AS (SELECT t_dat, cust, article_id, price, channel, count(*) AS n FROM {TX} GROUP BY ALL)
             SELECT t_dat, sum(CASE WHEN n > 1 THEN n - 1 ELSE 0 END) / sum(n) AS extra_share FROM g GROUP BY 1\"\"\")["extra_share"]
print(f"share of extra duplicate rows per day: min {daily.min():.2%}, median {daily.median():.2%}, max {daily.max():.2%} "
      f"(10th-90th pct {daily.quantile(.1):.2%}-{daily.quantile(.9):.2%})")
raw = con.execute(f\"\"\"SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM
          read_csv_auto('{(RAW / 'transactions_train.csv').as_posix()}', types={{'customer_id': 'VARCHAR', 'article_id': 'VARCHAR'}})))
          FROM read_csv_auto('{(RAW / 'transactions_train.csv').as_posix()}', types={{'customer_id': 'VARCHAR', 'article_id': 'VARCHAR'}})\"\"\").fetchone()[0]
print(f"same check on the raw CSV (real customer_id, unhashed): {raw:,} extra rows (Parquet: {extra:,})")"""),
    ("code", """mix = q(f\"\"\"WITH g AS (SELECT t_dat, cust, article_id, price, channel, count(*) AS n FROM {TX} GROUP BY ALL)
            SELECT a.product_group_name, sum(CASE WHEN n > 1 THEN n ELSE 0 END) AS dup_rows, sum(n) AS all_rows
            FROM g JOIN {ART} a USING (article_id) GROUP BY 1 ORDER BY dup_rows DESC LIMIT 8\"\"\")
mix["share_of_group_rows_duplicated"] = (mix["dup_rows"] / mix["all_rows"]).map("{:.1%}".format)
print(mix.to_string(index=False))
px = q(f\"\"\"WITH g AS (SELECT t_dat, cust, article_id, price, channel, count(*) AS n FROM {TX} GROUP BY ALL)
          SELECT CASE WHEN n > 1 THEN 'duplicated' ELSE 'single' END AS kind, median(price) AS median_price, count(*) AS groups
          FROM g GROUP BY 1\"\"\")
print(); print(px.to_string(index=False))"""),
    ("md", """**Reading:** the dataset has no quantity column, and every duplicate row has the same customer, day, article,
price *and* channel. The count of groups falls steeply with group size (2 ≫ 3 ≫ 4), with small bumps at even counts
(6, 8) that fit buying in pairs. The share of duplicate rows is spread over every day (printed range above) rather
than concentrated on a few days, which argues against a one-off logging fault, and the median price of duplicated
and single rows is the same. The raw CSV with the real customer_id gives the same count as the Parquet file, so it
is not caused by hashing. We cannot prove it without an order id or quantity field, but multiple units bought in one
transaction is the most consistent explanation. We treat each row as one unit sold: weekly `units = count(*)` keeps
them, and dropping them would undercount multi-buys."""),

    ("md", "### Prices: scaling and outliers"),
    ("code", """p = q(f"SELECT price FROM {TX} USING SAMPLE 2000000 ROWS (reservoir, 42)")["price"]
print(p.describe(percentiles=[.01, .5, .99]).to_string())
po = q(f\"\"\"WITH x AS (SELECT a.product_type_name, ln(t.price) AS lp FROM {TX} t JOIN {ART} a USING (article_id) WHERE t.price > 0),
           q AS (SELECT product_type_name, quantile_cont(lp, 0.25) AS q1, quantile_cont(lp, 0.75) AS q3 FROM x GROUP BY 1)
           SELECT count(*) AS rows, sum(CAST(lp > q3 + 3 * (q3 - q1) AS INT)) AS high, sum(CAST(lp < q1 - 3 * (q3 - q1) AS INT)) AS low
           FROM x JOIN q USING (product_type_name)\"\"\")
print(po.assign(high_share=po.high / po.rows, low_share=po.low / po.rows).to_string(index=False))
print("rows with price <= 0:", q(f"SELECT count(*) AS n FROM {TX} WHERE price <= 0")["n"][0])"""),
    ("md", """Prices are **scaled** by Kaggle (summary above: max ≈ 0.59, median ≈ 0.025), not in a currency. Ratios within the data are
meaningful (the model's `discount` = 1 − price / highest price seen), absolute levels are not. Outliers are flagged
on log price within each product type (beyond 3 × IQR). There are no zero or negative prices. Low outliers (0.26% of rows;
high outliers 0.04%) are kept: they are consistent with markdowns, and without a price list there is no way to
tell them from errors."""),

    ("md", "### Missing values: article descriptions and customer age"),
    ("code", """print(q(f"SELECT count(*) AS articles, sum(CAST(detail_desc IS NULL OR trim(detail_desc) = '' AS INT)) AS missing_detail_desc FROM {ART}")
      .assign(share=lambda d: (d.missing_detail_desc / d.articles).map("{:.2%}".format)).to_string(index=False))
print(q("SELECT count(*) AS customers, sum(CAST(age IS NULL AS INT)) AS missing_age FROM cust")
      .assign(share=lambda d: (d.missing_age / d.customers).map("{:.2%}".format)).to_string(index=False))"""),

    ("md", "### Articles and styles with no sales"),
    ("code", """ns = q(f\"\"\"WITH s AS (SELECT DISTINCT article_id FROM {TX})
          SELECT count(*) AS articles, sum(CAST(s.article_id IS NULL AS INT)) AS articles_no_sales,
                 count(DISTINCT a.product_code) AS styles,
                 count(DISTINCT a.product_code) FILTER (WHERE a.product_code NOT IN
                     (SELECT lpad(CAST(article_id // 1000 AS VARCHAR), 7, '0') FROM s)) AS styles_no_sales
          FROM {ART} a LEFT JOIN s USING (article_id)\"\"\")
print(ns.to_string(index=False))"""),

    ("md", "### Is product_code (our style id) stable across its articles?"),
    ("code", """print(q(f"SELECT count(*) AS articles, sum(CAST(article_id // 1000 <> CAST(product_code AS INT) AS INT)) AS id_mismatch FROM {ART}").to_string(index=False))
cols = ["prod_name", "product_type_name", "product_group_name", "garment_group_name", "index_group_name", "department_name", "colour_group_name"]
st = q(f"SELECT {', '.join(f'count(DISTINCT {c}) AS {c}' for c in cols)}, count(*) AS n_articles FROM {ART} GROUP BY product_code")
print(f"styles: {len(st):,} | articles per style: mean {st.n_articles.mean():.2f}, max {st.n_articles.max()}")
print("share of styles with more than one value:")
print((st[cols] > 1).mean().map("{:.2%}".format).to_string())"""),
    ("md", """`article_id // 1000 = product_code` for every article, so the style id is consistent. Colour varies within a
style by design (articles = colour variants). Product type and garment group are stable for almost all styles; where
a style has more than one, the model uses the most common value (`data._attribute_modes`)."""),

    ("md", "### Date coverage"),
    ("code", """d = q(f"SELECT t_dat, count(*) AS rows FROM {TX} GROUP BY 1 ORDER BY 1")
d["t_dat"] = pd.to_datetime(d["t_dat"])
full = pd.date_range(d.t_dat.min(), d.t_dat.max())
missing = full.difference(d.t_dat)
print(f"{d.t_dat.min():%Y-%m-%d} → {d.t_dat.max():%Y-%m-%d}: {len(full)} days, {len(d)} with data, missing: {list(missing.strftime('%Y-%m-%d'))}")
print("lowest-volume days:"); print(d.nsmallest(5, "rows").assign(t_dat=lambda x: x.t_dat.dt.strftime('%Y-%m-%d')).to_string(index=False))
print("highest-volume days:"); print(d.nlargest(5, "rows").assign(t_dat=lambda x: x.t_dat.dt.strftime('%Y-%m-%d')).to_string(index=False))"""),

    ("md", """## 3. Sampling note and the bias it introduces

- **Used in full:** all 31.8M transactions, all 105,542 articles and all 1.37M customers. No row sampling, except a
  2M-row reservoir sample (seed 42) for the price summary statistics; the price outlier counts use every row.
- **Images:** only the 9 reference photos of the 3 winning styles (3 colours each) were downloaded. The full image
  set is about 30 GB and the forecast does not use images.
- **Bias:** the forecast is image-free, so nothing in the ranking is biased by the photos. The concepts are, though:
  they reflect only the best-selling colours of 3 styles, so colours and styles that were not chosen never reach the
  design step. The image critic was also only checked against these 9 photos.
- **Data biases to keep in mind:** sales = demand only when items were in stock (no stock data); `customers.csv`
  age is self-reported and 1.2% missing; one customer-day is treated as one basket because there is no order id."""),
]


def build() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbformat.write(nb, NB)
    print(f"wrote {NB.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
