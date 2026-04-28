"""Small CLI wrapper for top-level project modules."""

from __future__ import annotations

import argparse

from modules.log_regression.baselineModel import main as log_regression_main
from modules.log_regression.fullDataPipeline import main as log_regression_pipeline_main
from modules.ranker.calc_stats import main as calc_stats_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a top-level VPR module.")
    parser.add_argument(
        "module",
        nargs="?",
        choices=["log_regression", "log_regression_pipeline", "ranking"],
        default="log_regression",
        help="Module to run. Defaults to the full log-regression stack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.module == "log_regression":
        print("Running log regression module with data pipeline")
        log_regression_pipeline_main()
        log_regression_main()
    elif args.module == "log_regression_pipeline":
        print("Running log regression data pipeline only")
        log_regression_pipeline_main()
    elif args.module == "ranking":
        print("Running ranking module")
        calc_stats_main()


if __name__ == "__main__":
    main()
