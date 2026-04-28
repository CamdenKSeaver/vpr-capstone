from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import requests


DEFAULT_SEASON = os.environ.get("VPR_SEASON", "2025-2026")
DATA_CSV_BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv"
)
DATA_CSV_LISTING_URL = (
    "https://api.github.com/repos/JeffreyRStevens/ncaavolleyballr/contents/data-csv"
)
SOURCE_FILE_TEMPLATES = {
    "players": "wvb_playerseason_div1_{year}.csv",
    "player_match": "wvb_playermatch_div1_{year}.csv",
    "team_match": "wvb_teammatch_div1_{year}.csv",
    "pbp": "wvb_pbp_div1_{year}.csv",
}
REMOTE_FILE_PATTERNS = {
    key: re.compile(template.replace("{year}", r"(\d{4})"))
    for key, template in SOURCE_FILE_TEMPLATES.items()
}


def add_season_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--season", default=None)
    parser.add_argument("--seasons", nargs="+", default=None)
    parser.add_argument(
        "--season-range",
        nargs=2,
        metavar=("START_SEASON", "END_SEASON"),
        default=None,
    )
    parser.add_argument("--all-seasons", action="store_true")
    parser.add_argument(
        "--download-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing source files from the upstream ncaavolleyballr data-csv page.",
    )
    return parser


def normalize_season_text(season: str) -> str:
    cleaned = str(season).strip()
    match = re.fullmatch(r"(\d{4})-(\d{4})", cleaned)
    if not match:
        raise ValueError(
            f"Season '{season}' must look like YYYY-YYYY, for example 2025-2026."
        )
    start = int(match.group(1))
    end = int(match.group(2))
    if end != start + 1:
        raise ValueError(f"Season '{season}' is not a consecutive year range.")
    return f"{start}-{end}"


def season_to_source_year(season: str) -> str:
    return normalize_season_text(season).split("-")[0]


def source_year_to_season(year: str | int) -> str:
    start = int(year)
    return f"{start}-{start + 1}"


def sort_seasons(seasons: list[str]) -> list[str]:
    return sorted(set(seasons), key=lambda s: int(season_to_source_year(s)))


def expand_season_range(start_season: str, end_season: str) -> list[str]:
    start_year = int(season_to_source_year(start_season))
    end_year = int(season_to_source_year(end_season))
    if end_year < start_year:
        raise ValueError("Season range end must be the same as or after the start.")
    return [source_year_to_season(year) for year in range(start_year, end_year + 1)]


def _expand_season_tokens(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return []
    expanded: list[str] = []
    for token in tokens:
        parts = [part.strip() for part in str(token).split(",") if part.strip()]
        expanded.extend(parts)
    return expanded


def list_remote_available_seasons(timeout: int = 30) -> list[str]:
    response = requests.get(DATA_CSV_LISTING_URL, timeout=timeout)
    response.raise_for_status()
    names = [
        item["name"]
        for item in response.json()
        if isinstance(item, dict) and "name" in item
    ]

    season_sets: dict[str, set[str]] = {key: set() for key in SOURCE_FILE_TEMPLATES}
    for name in names:
        for key, pattern in REMOTE_FILE_PATTERNS.items():
            match = pattern.fullmatch(name)
            if match:
                season_sets[key].add(match.group(1))

    common_years = set.intersection(*season_sets.values())
    return sort_seasons([source_year_to_season(year) for year in common_years])


def list_local_available_seasons(data_dir: Path) -> list[str]:
    seasons: set[str] = set()
    for path in data_dir.glob("wvb_playerseason_div1_*.csv"):
        match = re.fullmatch(r"wvb_playerseason_div1_(\d{4})\.csv", path.name)
        if match:
            seasons.add(source_year_to_season(match.group(1)))

    aggregate_players = data_dir / "all_players.csv"
    if aggregate_players.exists():
        try:
            player_df = pd.read_csv(aggregate_players, usecols=["Season"], low_memory=False)
            seasons.update(
                normalize_season_text(season)
                for season in player_df["Season"].dropna().astype(str).unique().tolist()
            )
        except Exception:
            pass

    return sort_seasons(list(seasons))


def resolve_requested_seasons(
    *,
    season: str | None,
    seasons: list[str] | None,
    season_range: list[str] | None,
    all_seasons: bool,
    data_dir: Path,
    download_missing: bool,
) -> list[str]:
    explicit = _expand_season_tokens(seasons)
    if all_seasons:
        available: list[str] = []
        if download_missing:
            try:
                available = list_remote_available_seasons()
            except Exception:
                available = []
        if not available:
            available = list_local_available_seasons(data_dir)
        if not available:
            raise FileNotFoundError(
                "Could not discover any available seasons from the remote source or local cache."
            )
        return available

    if explicit:
        return [normalize_season_text(item) for item in explicit]
    if season_range:
        start_season, end_season = season_range
        return expand_season_range(start_season, end_season)
    if season:
        return [normalize_season_text(season)]
    return [normalize_season_text(DEFAULT_SEASON)]


def build_remote_url(filename: str) -> str:
    return f"{DATA_CSV_BASE_URL}/{filename}"


def stream_download(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)
    return destination


def ensure_remote_source_file(data_dir: Path, filename: str, timeout: int = 60) -> Path:
    path = data_dir / filename
    if path.exists():
        return path
    url = build_remote_url(filename)
    print(f"Downloading missing source file: {filename}")
    return stream_download(url, path, timeout=timeout)


def ensure_season_source_files(
    data_dir: Path,
    season: str,
    *,
    include_pbp: bool = True,
    download_missing: bool = True,
) -> dict[str, Path]:
    year = season_to_source_year(season)
    needed_keys = ["players", "player_match", "team_match"]
    if include_pbp:
        needed_keys.append("pbp")

    files: dict[str, Path] = {}
    for key in needed_keys:
        filename = SOURCE_FILE_TEMPLATES[key].format(year=year)
        path = data_dir / filename
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(f"Missing required source file: {path}")
            path = ensure_remote_source_file(data_dir, filename)
        files[key] = path
    return files
