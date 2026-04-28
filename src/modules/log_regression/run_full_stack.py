from __future__ import annotations
import sys
from scmrepo.git import Git
from pathlib import Path

ROOT = Path(Git(root_dir=".").root_dir) / "src" / "modules"
sys.path.append(str(ROOT))

from log_regression.baselineModel import combine_model_outputs
from log_regression.baselineModel import run_for_season as run_model_for_season
from log_regression.fullDataPipeline import run_for_season as run_preprocess_for_season
from log_regression.season_support import add_season_args, resolve_requested_seasons
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full volleyball ranking stack for one or more seasons."
    )
    add_season_args(parser)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--save-event-table",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-train-rallies",
        type=int,
        default=500_000,
        help="Maximum train rallies for the win-probability fit. Use 0 for all.",
    )
    parser.add_argument("--shrinkage-tau", type=float, default=250.0)
    parser.add_argument("--min-rallies", type=int, default=200)
    parser.add_argument(
        "--wp-model",
        choices=["logistic", "mlp"],
        default="mlp",
        help="Model family for rally win probability. Defaults to mlp.",
    )
    parser.add_argument(
        "--combine-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.repo_root or Path(Git(root_dir=".").root_dir)
    data_dir = package_root / "Data" / "original"
    seasons = resolve_requested_seasons(
        season=args.season,
        seasons=args.seasons,
        season_range=args.season_range,
        all_seasons=args.all_seasons,
        data_dir=data_dir,
        download_missing=args.download_missing,
    )
    max_train_rallies = None if args.max_train_rallies == 0 else args.max_train_rallies

    for season in seasons:
        run_preprocess_for_season(
            season,
            repo_root=package_root,
            save_event_table=args.save_event_table,
            download_missing=args.download_missing,
        )
        run_model_for_season(
            season,
            repo_root=package_root,
            max_train_rallies=max_train_rallies,
            shrinkage_tau=args.shrinkage_tau,
            min_rallies=args.min_rallies,
            wp_model_type=args.wp_model,
            build_missing_preprocessed=False,
            download_missing=args.download_missing,
        )

    if args.combine_outputs:
        combine_model_outputs(package_root / "Data" / "model_outputs", seasons)


if __name__ == "__main__":
    main()
