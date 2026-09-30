"""Convert the Kaggle H&M CSVs into the Parquet layout the pipeline reads.

  kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv -p raw
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv -p raw
  python scripts/convert_data.py --raw raw --out data        # then set HM_DATA_DIR=data (the default)

Writes <out>/tx/part_0.parquet (t_dat DATE, cust UBIGINT = 64-bit hash of customer_id, article_id INTEGER,
price FLOAT, channel TINYINT) and <out>/articles.parquet (Kaggle columns; product_code as a 7-char string).
Uses DuckDB with a 2 GB memory cap, so the 3.5 GB transactions CSV never has to fit in RAM.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import duckdb


def _csv(raw: Path, name: str) -> str:
    """Path to <name>.csv, unzipping <name>.csv.zip (Kaggle's download format) if needed."""
    csv = raw / f"{name}.csv"
    if not csv.exists() and (raw / f"{name}.csv.zip").exists():
        with zipfile.ZipFile(raw / f"{name}.csv.zip") as z:
            z.extractall(raw)
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — download it from Kaggle first (see the module docstring).")
    return csv.as_posix()


def convert(raw: Path, out: Path) -> None:
    """Write tx/part_0.parquet and articles.parquet under ``out``."""
    (out / "tx").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
        COPY (
            SELECT CAST(t_dat AS DATE)               AS t_dat,
                   CAST(hash(customer_id) AS UBIGINT) AS cust,
                   CAST(article_id AS INTEGER)        AS article_id,
                   CAST(price AS FLOAT)               AS price,
                   CAST(sales_channel_id AS TINYINT)  AS channel
            FROM read_csv('{_csv(raw, "transactions_train")}', header = true,
                          types = {{'article_id': 'VARCHAR', 'customer_id': 'VARCHAR'}})
        ) TO '{(out / "tx" / "part_0.parquet").as_posix()}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (
            SELECT * REPLACE (CAST(article_id AS INTEGER) AS article_id,
                              lpad(product_code, 7, '0') AS product_code)
            FROM read_csv('{_csv(raw, "articles")}', header = true,
                          types = {{'article_id': 'VARCHAR', 'product_code': 'VARCHAR',
                                    'colour_group_code': 'VARCHAR', 'index_code': 'VARCHAR'}})
        ) TO '{(out / "articles.parquet").as_posix()}' (FORMAT parquet)
    """)
    n, d0, d1 = con.execute(f"SELECT count(*), min(t_dat), max(t_dat) FROM read_parquet('{(out / 'tx').as_posix()}/*.parquet')").fetchone()
    a = con.execute(f"SELECT count(*) FROM read_parquet('{(out / 'articles.parquet').as_posix()}')").fetchone()[0]
    print(f"transactions: {n:,} rows ({d0} → {d1}) | articles: {a:,} rows → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder with transactions_train.csv(.zip), articles.csv(.zip)")
    ap.add_argument("--out", type=Path, default=Path("data"), help="output folder (= HM_DATA_DIR)")
    a = ap.parse_args()
    convert(a.raw, a.out)
