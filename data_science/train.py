"""Training pipeline entry point.

  python -m data_science.train                    # features cache + winner classifiers + final scores
  python -m data_science.train --with-regressor   # also retrain the regressor and re-select the top-3

Stages, in order:
1. data_science.features   weekly style cache + leakage-free train/validation snapshots (outputs/cache/)
2. data_science.model      regressor vs ranker, rolling backtest, eval_table.md, movers.md      (--with-regressor)
3. data_science.select     final regressor, top-3 evidence, seasonal comparison                (--with-regressor)
4. data_science.classify   winner classifiers (top 1% and top 0.1%), calibration, final scores

Stages 2-3 rewrite the published evaluation, models and evidence files, and a different OS/Python can shift
the numbers slightly, so they only run when asked for. Then run ``python -m data_science.predict``.
"""
from __future__ import annotations

import argparse
import runpy


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the training pipeline")
    ap.add_argument("--with-regressor", action="store_true",
                    help="also rerun data_science.model and data_science.select (rewrites published outputs)")
    args = ap.parse_args()

    runpy.run_module("data_science.features", run_name="__main__")
    if args.with_regressor:
        from data_science import model, select
        model.main()
        select.main()
    from data_science import classify
    classify.run(classify.WINNER_SHARE)
    classify.run(classify.STRICT_SHARE)
    classify.combine()


if __name__ == "__main__":
    main()
