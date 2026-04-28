"""Build cleaned season tables from the raw NCAA women's volleyball source files.

This module turns roster, match, and play-by-play downloads into the
preprocessed CSVs used by the ranking model.
"""
from __future__ import annotations
import sys
from scmrepo.git import Git
from pathlib import Path

ROOT = Path(Git(root_dir=".").root_dir) / "src" / "modules"
sys.path.append(str(ROOT))
import argparse
import re
import numpy as np
import pandas as pd
from log_regression.season_support import (
    DEFAULT_SEASON,
    add_season_args,
    ensure_season_source_files,
    resolve_requested_seasons,
)

# Keep only the play-by-play columns that feed the later joins and rally logic.
PBP_DESIRED_COLS = [
    "date",
    "contestid",
    "set",
    "away_team",
    "home_team",
    "score",
    "rally",
    "rally_event",
    "team",
    "event",
    "player",
    "description",
]

PBP_DTYPE_MAP = {
    "date": "string",
    "contestid": "string",
    "set": "string",
    "away_team": "string",
    "home_team": "string",
    "score": "string",
    "rally": "string",
    "rally_event": "string",
    "team": "string",
    "event": "string",
    "player": "string",
    "description": "string",
}

ROLE_MAP = {
    # Position labels vary by source file, so normalize them into the ranking
    # buckets first and let the model layer make any final role adjustments.
    "s": "setter",
    "setter": "setter",
    "mb": "middle",
    "mh": "middle",
    "m": "middle",
    "middle blocker": "middle",
    "oh": "outside",
    "ls": "outside",
    "os": "outside",
    "outside hitter": "outside",
    "rs": "opposite",
    "opp": "opposite",
    "opposite": "opposite",
    "right side": "opposite",
    "l": "libero_ds",
    "ds": "libero_ds",
    "l/ds": "libero_ds",
    "libero": "libero_ds",
}

MANUAL_TEAM_ALIASES = {
    # Manual fixes for school-name variants that appear across rosters, match
    # tables, and play-by-play feeds.
    "app state": "app state",
    "arkansas state": "arkansas st.",
    "cal baptist": "california baptist",
    "coastal carolina": "coastal carolina",
    "ga southern": "ga. southern",
    "georgia southern": "ga. southern",
    "georgia state": "georgia st.",
    "la.-monroe": "ulm",
    "louisiana monroe": "ulm",
    "loyola marymount": "lmu (ca)",
    "miami ohio": "miami (oh)",
    "miami (ohio)": "miami (oh)",
    "saint marys": "saint marys (ca)",
    "saint marys ca": "saint marys (ca)",
    "saint peters": "saint peters",
    "st thomas": "st. thomas (mn)",
    "st. thomas": "st. thomas (mn)",
    "texas rio grande valley": "utrgv",
    "ut rio grande valley": "utrgv",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so the different source files can share logic."""
    rename_map = {
        c: re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip(
            "_"
        )
        for c in df.columns
    }
    df = df.rename(columns=rename_map).copy()
    alias_map = {
        "totalattacks": "total_attacks",
        "hitpct": "hit_pct",
        "blocksolos": "block_solos",
        "blockassists": "block_assists",
        "opponent_team": "opponent",
    }
    alias_map = {k: v for k, v in alias_map.items() if k in df.columns}
    return df.rename(columns=alias_map)


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate columns can happen after cleaning names, so keep the first real value."""
    df = df.copy()
    cols = pd.Series(df.columns)
    dup_names = cols[cols.duplicated(keep=False)].unique().tolist()
    for name in dup_names:
        # If two cleaned columns collapse to the same name, keep the left-most
        # non-null value on each row.
        same = df.loc[:, df.columns == name]
        merged = same.bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=name)
        df[name] = merged
    return df


def clean_series(s: pd.Series) -> pd.Series:
    # Normalize names hard enough that roster, match, and play-by-play tables
    # can join reliably across inconsistent source formatting.
    s = s.fillna("").astype("string")
    s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    s = s.str.lower().str.replace("&", " and ", regex=False)
    s = s.str.replace(r"[\u2018\u2019\u201c\u201d'`]", "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s/@().,#:-]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def clean_team_series(s: pd.Series) -> pd.Series:
    s = clean_series(s)
    s = s.str.replace(r"\bvs\.?\b", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def strip_opponent_noise(s: pd.Series) -> pd.Series:
    s = clean_team_series(s)
    # Opponent fields often include rankings, venues, and tournament labels that
    # are useful for display but noisy for team matching.
    s = s.str.replace(r"^@\s*", "", regex=True)
    s = s.str.replace(r"\b(at)\b\s+", "", regex=True)
    s = s.str.replace(r"#\d+", " ", regex=True)
    s = s.str.replace(r",.*$", "", regex=True)
    s = s.str.replace(
        r"\b(classic|challenge|invitational|tournament|regional|showcase)\b.*$",
        "",
        regex=True,
    )
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def canonicalize_vs_known_teams(
    raw_series: pd.Series,
    canonical_teams: list[str],
    manual_aliases: dict[str, str] | None = None,
) -> pd.Series:
    """Clean team names against the teams we already know."""
    # Prefer exact and manual matches first. It is safer to leave a school
    # unresolved than to force a bad canonical match.
    manual_aliases = manual_aliases or {}
    raw = strip_opponent_noise(raw_series)
    canonical_set = set(canonical_teams)

    result = raw.map(manual_aliases)
    result = result.fillna(raw.where(raw.isin(canonical_set)))

    unmatched = raw[result.isna()].dropna().unique().tolist()
    fallback: dict[str, str | pd._libs.missing.NAType] = {}
    for val in unmatched:
        compact = re.sub(r"[^a-z0-9]", "", val)
        hits = []
        for team in canonical_teams:
            team_compact = re.sub(r"[^a-z0-9]", "", team)
            if team_compact == compact:
                hits.append(team)
            elif team.startswith(f"{val} ") or team.startswith(f"{val}("):
                hits.append(team)
            elif re.search(rf"(?<!\w){re.escape(team)}(?!\w)", val):
                hits.append(team)
        fallback[val] = max(set(hits), key=len) if len(set(hits)) == 1 else pd.NA

    result = result.fillna(raw.map(fallback))
    return result.fillna(raw)


def load_csv_flexible(
    path_or_url: Path | str, desired_cols=None, dtype_map=None
) -> pd.DataFrame:
    header = pd.read_csv(path_or_url, nrows=0)
    cols = header.columns.tolist()
    usecols = cols if desired_cols is None else [c for c in cols if c in desired_cols]
    dtype_map = dtype_map or {}
    use_dtypes = {k: v for k, v in dtype_map.items() if k in usecols}
    return pd.read_csv(path_or_url, usecols=usecols, dtype=use_dtypes, low_memory=False)


def save_csv(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    print(f"saved: {path} ({len(df):,} rows)")
    return path


def zscore(s: pd.Series) -> pd.Series:
    # normal zscore helper
    # if a column is constant just return 0s instead of breaking everything
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
    return (s - s.mean()) / sd


def canonical_event_family(
    event_text: pd.Series, description_text: pd.Series
) -> pd.Series:
    # turn the messy pbp text into a smaller list of event types
    # not perfect, but good enough to give model.py a consistent language
    evt = event_text.fillna("").astype("string")
    desc = description_text.fillna("").astype("string")
    family = pd.Series(pd.NA, index=evt.index, dtype="string")

    rules = [
        ("kill", r"first ball kill", r"first ball kill"),
        ("ace", r"service ace|^ace$", r"service ace| ace by"),
        ("service_error", r"service error", r"service error"),
        ("reception_error", r"reception error", r"reception error"),
        (
            "set_error",
            r"set error|ball handling error",
            r"set error|ball handling error",
        ),
        ("attack_error", r"attack error", r"attack error"),
        ("block_error", r"block error", r"block error"),
        ("dig_error", r"dig error", r"dig error"),
        ("serve", r"^serve$| serves", r" serves$"),
        ("reception", r"^reception$|reception by", r"reception by"),
        ("set", r"^set$|set by", r"set by"),
        ("attack", r"^attack$", r"attack by"),
        ("kill", r"^kill$| kill by", r" kill by"),
        ("dig", r"^dig$|dig by", r"dig by"),
        ("block", r"^block$|block by", r"block by"),
    ]

    for label, event_pattern, desc_pattern in rules:
        mask = family.isna() & (
            evt.str.contains(event_pattern, regex=True, na=False)
            | desc.str.contains(desc_pattern, regex=True, na=False)
        )
        family = family.mask(mask, label)

    admin_mask = evt.str.contains(
        r"timeout|challenge|sanction|sub|substitution|media timeout|\+",
        regex=True,
        na=False,
    ) | desc.str.contains(
        r"timeout|challenge|sanction|sub|substitution|media timeout",
        regex=True,
        na=False,
    )
    family = family.mask(family.isna() & admin_mask, "admin")
    return family.fillna("admin")


def load_season_source_files(
    data_dir: Path,
    season: str,
    *,
    download_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # use the upstream season files directly
    # this avoids relying on one local master file to cover every season forever
    source_files = ensure_season_source_files(
        data_dir,
        season,
        include_pbp=False,
        download_missing=download_missing,
    )
    players = coalesce_duplicate_columns(
        standardize_columns(pd.read_csv(source_files["players"], low_memory=False))
    )
    player_match = coalesce_duplicate_columns(
        standardize_columns(pd.read_csv(source_files["player_match"], low_memory=False))
    )
    team_match = coalesce_duplicate_columns(
        standardize_columns(pd.read_csv(source_files["team_match"], low_memory=False))
    )

    if "player" in players.columns:
        bad_player_labels = {"team", "totals", "opponent totals", "opponent"}
        players = players.loc[
            ~clean_series(players["player"]).isin(bad_player_labels)
        ].copy()

    player_match = player_match.drop_duplicates().copy()
    return players, player_match, team_match


def build_player_master(players: pd.DataFrame) -> pd.DataFrame:
    # main player identity table
    # most later joins are basically team_clean + player_clean
    players = players.copy()
    players["player_clean"] = clean_series(players["player"])
    players["team_clean"] = clean_team_series(players["team"])
    players["pos_clean"] = (
        clean_series(players["pos"]) if "pos" in players.columns else ""
    )
    players["role_family"] = players["pos_clean"].map(ROLE_MAP).fillna("unknown")
    players["number_clean"] = pd.to_numeric(
        players.get("number"), errors="coerce"
    ).astype("Int64")
    players["player_uid"] = (
        players["season"].astype("string")
        + "::"
        + players["team_clean"].astype("string")
        + "::"
        + players["player_clean"].astype("string")
        + "::"
        + players["number_clean"].astype("string")
    )

    keep = [
        "season",
        "teamid",
        "team",
        "team_clean",
        "conference",
        "number",
        "number_clean",
        "player",
        "player_clean",
        "pos",
        "role_family",
        "player_uid",
    ]
    for col in keep:
        if col not in players.columns:
            players[col] = pd.NA
    return players[keep].drop_duplicates().reset_index(drop=True)


def prepare_match_tables(
    player_match: pd.DataFrame,
    team_match: pd.DataFrame,
    player_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # clean match tables and attach roster roles/player ids
    # also builds the known team list for the opponent cleanup
    player_match = player_match.copy()
    team_match = team_match.copy()

    player_match["player_clean"] = clean_series(player_match["player"])
    player_match["team_clean"] = clean_team_series(player_match["team"])
    team_match["team_clean"] = clean_team_series(team_match["team"])

    canonical_teams = sorted(
        set(player_master["team_clean"].dropna().unique()).union(
            set(team_match["team_clean"].dropna().unique())
        )
    )

    player_match["opp_clean"] = canonicalize_vs_known_teams(
        player_match["opponent"], canonical_teams, MANUAL_TEAM_ALIASES
    )
    team_match["opp_clean"] = canonicalize_vs_known_teams(
        team_match["opponent"], canonical_teams, MANUAL_TEAM_ALIASES
    )

    role_lookup = player_master[
        ["season", "team_clean", "player_clean", "role_family", "player_uid"]
    ].drop_duplicates()
    player_match = player_match.merge(
        role_lookup,
        on=["season", "team_clean", "player_clean"],
        how="left",
    )

    if "contestid" in player_match.columns:
        player_match["contestid"] = pd.to_numeric(
            player_match["contestid"], errors="coerce"
        ).astype("Int64")
    else:
        player_match["contestid"] = pd.Series(
            pd.NA, index=player_match.index, dtype="Int64"
        )

    return player_match, team_match, pd.DataFrame({"team_clean": canonical_teams})


def build_team_strength(
    team_match: pd.DataFrame, player_master: pd.DataFrame
) -> pd.DataFrame:
    # simple team strength, not some full elo thing
    # just enough to control for good/bad teams later
    team_match = team_match.copy()
    for col in [
        "hit_pct",
        "s",
        "kills",
        "errors",
        "assists",
        "aces",
        "serr",
        "digs",
        "pts",
    ]:
        if col not in team_match.columns:
            team_match[col] = np.nan

    result_parts = (
        team_match["result"]
        .fillna("")
        .astype("string")
        .str.extract(r"^\s*([WL])\s*(\d+)\s*-\s*(\d+)\s*$")
    )
    team_match["match_win"] = result_parts[0].fillna("").eq("W").astype("int8")
    team_match["sets_for"] = (
        pd.to_numeric(result_parts[1], errors="coerce").fillna(0).astype("int16")
    )
    team_match["sets_against"] = (
        pd.to_numeric(result_parts[2], errors="coerce").fillna(0).astype("int16")
    )
    team_match["hit_pct"] = pd.to_numeric(team_match["hit_pct"], errors="coerce")
    team_match["pts"] = pd.to_numeric(team_match["pts"], errors="coerce")

    team_strength = (
        team_match.groupby(["season", "team_clean"], observed=True)
        .agg(
            matches=("team_clean", "size"),
            win_pct=("match_win", "mean"),
            sets_for=("sets_for", "sum"),
            sets_against=("sets_against", "sum"),
            mean_hit_pct=("hit_pct", "mean"),
            mean_pts=("pts", "mean"),
        )
        .reset_index()
    )
    team_strength["set_diff_per_match"] = (
        (team_strength["sets_for"] - team_strength["sets_against"])
        / team_strength["matches"].replace(0, np.nan)
    ).fillna(0)
    team_strength["team_strength_index"] = (
        0.50 * zscore(team_strength["win_pct"])
        + 0.35 * zscore(team_strength["set_diff_per_match"])
        + 0.15
        * zscore(
            team_strength["mean_hit_pct"].fillna(team_strength["mean_hit_pct"].median())
        )
    )
    team_names = (
        player_master[["team_clean", "team"]]
        .drop_duplicates("team_clean")
        .rename(columns={"team": "team_display"})
    )
    return team_strength.merge(team_names, on="team_clean", how="left")


def coerce_pbp_contest_ids(pbp: pd.DataFrame) -> pd.Series:
    # newer pbp files already ship with contest ids
    raw_ids = pd.to_numeric(
        pbp.get("contestid", pd.Series(pd.NA, index=pbp.index, dtype="string")),
        errors="coerce",
    )
    if raw_ids.notna().all():
        return raw_ids.astype("Int64")

    # older pbp files do not, so rebuild stable match blocks from row order
    key_frame = pd.DataFrame(index=pbp.index)
    for col in ["date", "home_team", "away_team"]:
        vals = pbp.get(col, pd.Series("", index=pbp.index, dtype="string"))
        key_frame[col] = (
            vals.fillna("").astype("string").str.strip().replace("", pd.NA).ffill()
        )

    set_numbers = pd.to_numeric(pbp.get("set"), errors="coerce")
    key_changed = (key_frame != key_frame.shift()).any(axis=1)
    set_reset = set_numbers.lt(set_numbers.shift()) & set_numbers.notna()
    new_contest = (key_changed | set_reset).fillna(False)
    if len(new_contest):
        new_contest.iloc[0] = True

    synthetic_ids = new_contest.cumsum().astype("Int64")
    if raw_ids.notna().any():
        block_lookup = (
            pd.DataFrame({"synthetic_id": synthetic_ids, "contestid": raw_ids})
            .dropna(subset=["contestid"])
            .drop_duplicates("synthetic_id")
            .set_index("synthetic_id")["contestid"]
        )
        return synthetic_ids.map(block_lookup).fillna(synthetic_ids).astype("Int64")
    return synthetic_ids


def load_and_clean_pbp(
    data_dir: Path,
    player_master: pd.DataFrame,
    canonical_teams: list[str],
    season: str,
    *,
    download_missing: bool = True,
) -> pd.DataFrame:
    # this is the slow part
    # pbp is big, so dont do random stuff here or it gets hard to debug
    source_files = ensure_season_source_files(
        data_dir,
        season,
        include_pbp=True,
        download_missing=download_missing,
    )
    pbp_path = source_files["pbp"]

    print(f"loading play-by-play: {pbp_path}")
    pbp_raw = load_csv_flexible(pbp_path, PBP_DESIRED_COLS, PBP_DTYPE_MAP)
    pbp = pbp_raw.drop_duplicates().copy()
    del pbp_raw

    pbp = standardize_columns(pbp)
    pbp["season"] = season
    pbp["file_row"] = np.arange(len(pbp), dtype="int64")
    pbp["set"] = pd.to_numeric(pbp["set"], errors="coerce").astype("Int16")
    pbp["contestid"] = coerce_pbp_contest_ids(pbp)
    pbp = pbp.dropna(subset=["contestid", "set"]).copy()

    for col in [
        "team",
        "home_team",
        "away_team",
        "player",
        "event",
        "description",
        "score",
        "date",
    ]:
        if col not in pbp.columns:
            pbp[col] = pd.Series("", index=pbp.index, dtype="string")

    pbp["team_clean"] = canonicalize_vs_known_teams(
        pbp["team"], canonical_teams, MANUAL_TEAM_ALIASES
    )
    pbp["home_clean"] = canonicalize_vs_known_teams(
        pbp["home_team"], canonical_teams, MANUAL_TEAM_ALIASES
    )
    pbp["away_clean"] = canonicalize_vs_known_teams(
        pbp["away_team"], canonical_teams, MANUAL_TEAM_ALIASES
    )
    pbp["player_clean"] = clean_series(pbp["player"])
    pbp["event_text"] = clean_series(pbp["event"])
    pbp["description_text"] = clean_series(pbp["description"])

    sort_cols = ["contestid", "set", "file_row"]
    pbp = pbp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    roster_key = player_master[["player_clean", "team_clean"]].drop_duplicates()
    # figure out who the actor actually plays for
    # raw team column gets weird on errors so roster lookup gets first shot
    home_lookup = roster_key.rename(columns={"team_clean": "home_clean"}).assign(
        player_on_home=True
    )
    away_lookup = roster_key.rename(columns={"team_clean": "away_clean"}).assign(
        player_on_away=True
    )
    pbp = pbp.merge(home_lookup, on=["player_clean", "home_clean"], how="left")
    pbp = pbp.merge(away_lookup, on=["player_clean", "away_clean"], how="left")
    pbp["player_on_home"] = pbp["player_on_home"].eq(True)
    pbp["player_on_away"] = pbp["player_on_away"].eq(True)
    pbp["team_matches_home"] = pbp["team_clean"].eq(pbp["home_clean"])
    pbp["team_matches_away"] = pbp["team_clean"].eq(pbp["away_clean"])

    pbp["actor_team"] = np.select(
        [
            pbp["player_on_home"] & ~pbp["player_on_away"],
            pbp["player_on_away"] & ~pbp["player_on_home"],
            pbp["team_matches_home"],
            pbp["team_matches_away"],
        ],
        [pbp["home_clean"], pbp["away_clean"], pbp["home_clean"], pbp["away_clean"]],
        default=pd.NA,
    )
    pbp["actor_side"] = np.select(
        [
            pbp["actor_team"].eq(pbp["home_clean"]),
            pbp["actor_team"].eq(pbp["away_clean"]),
        ],
        ["home", "away"],
        default="unknown",
    )
    pbp["actor_team_source"] = np.select(
        [
            pbp["player_on_home"] & ~pbp["player_on_away"],
            pbp["player_on_away"] & ~pbp["player_on_home"],
            pbp["team_matches_home"] | pbp["team_matches_away"],
        ],
        ["roster_home", "roster_away", "team_column"],
        default="unresolved",
    )

    score_parts = (
        pbp["score"].astype("string").str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    )
    # scores are how I rebuild rallies
    # weird score rows stay in, but they get flags so we can see them later
    pbp["away_score_raw"] = pd.to_numeric(score_parts[0], errors="coerce")
    pbp["home_score_raw"] = pd.to_numeric(score_parts[1], errors="coerce")
    pbp["score_is_numeric"] = (
        pbp["away_score_raw"].notna() & pbp["home_score_raw"].notna()
    )
    group_set = pbp.groupby(["contestid", "set"], sort=False, observed=True)
    pbp[["away_score_ffill", "home_score_ffill"]] = group_set[
        ["away_score_raw", "home_score_raw"]
    ].ffill()
    pbp[["away_score_ffill", "home_score_ffill"]] = pbp[
        ["away_score_ffill", "home_score_ffill"]
    ].fillna(0)
    pbp["d_away"] = group_set["away_score_ffill"].diff().fillna(0)
    pbp["d_home"] = group_set["home_score_ffill"].diff().fillna(0)
    pbp["score_backward"] = (pbp["d_away"] < 0) | (pbp["d_home"] < 0)
    pbp["multi_point_jump"] = (pbp["d_away"].abs() + pbp["d_home"].abs()) > 1
    pbp["one_point_increment"] = (
        (pbp["d_away"] >= 0)
        & (pbp["d_home"] >= 0)
        & ((pbp["d_away"] + pbp["d_home"]) == 1)
    )

    pbp["event_family"] = canonical_event_family(
        pbp["event_text"], pbp["description_text"]
    )
    pbp["is_first_ball_kill_text"] = pbp["event_text"].str.contains(
        r"first ball kill", na=False
    ) | pbp["description_text"].str.contains(r"first ball kill", na=False)
    pbp["is_action_row"] = pbp["event_family"].ne("admin")

    prev_end = pbp.groupby(["contestid", "set"], sort=False, observed=True)[
        "one_point_increment"
    ].shift(fill_value=False)
    pbp["rally_id"] = (
        prev_end.groupby([pbp["contestid"], pbp["set"]], sort=False)
        .cumsum()
        .astype("int32")
        + 1
    )
    group_rally = ["contestid", "set", "rally_id"]
    pbp["row_in_rally"] = (
        pbp.groupby(group_rally, sort=False, observed=True).cumcount().astype("int16")
        + 1
    )
    pbp["action_seq"] = (
        pbp["is_action_row"]
        .astype("int16")
        .groupby([pbp["contestid"], pbp["set"], pbp["rally_id"]], sort=False)
        .cumsum()
        .astype("int16")
    )

    role_lookup = player_master[
        ["season", "team_clean", "player_clean", "player_uid", "role_family"]
    ].drop_duplicates()
    pbp = pbp.merge(
        role_lookup.rename(columns={"team_clean": "actor_team"}),
        on=["season", "actor_team", "player_clean"],
        how="left",
    )

    return pbp


def build_rally_tables(
    pbp: pd.DataFrame, season: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # turn all the individual event rows into rally rows
    # event_table keeps the touch level data, rally_table is for win prob later
    group_cols = ["contestid", "set", "rally_id"]

    first_serve = (
        # first real serve row is the server for the rally
        # this skips past random admin stuff if it appears first
        pbp.loc[pbp["event_family"].eq("serve")]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[
            group_cols + ["file_row", "actor_team", "actor_side", "player_clean"]
        ]
        .rename(
            columns={
                "file_row": "serve_row",
                "actor_team": "serve_team",
                "actor_side": "server_side",
                "player_clean": "server_player_clean",
            }
        )
    )

    last_action = (
        pbp.loc[pbp["is_action_row"]]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .tail(1)[
            group_cols
            + ["file_row", "event_family", "actor_team", "actor_side", "player_clean"]
        ]
        .rename(
            columns={
                "file_row": "terminal_row",
                "event_family": "terminal_event_family",
                "actor_team": "terminal_actor_team",
                "actor_side": "terminal_actor_side",
                "player_clean": "terminal_player_clean",
            }
        )
    )

    rally_table = (
        pbp.groupby(group_cols, sort=False, observed=True)
        .agg(
            season=("season", "first"),
            date=("date", "first"),
            away_team=("away_team", "first"),
            home_team=("home_team", "first"),
            away_clean=("away_clean", "first"),
            home_clean=("home_clean", "first"),
            score_away_start=("away_score_ffill", "first"),
            score_home_start=("home_score_ffill", "first"),
            score_away_end=("away_score_ffill", "last"),
            score_home_end=("home_score_ffill", "last"),
            rally_rows=("file_row", "size"),
            action_rows=("is_action_row", "sum"),
            unresolved_actor_rows=(
                "actor_team_source",
                lambda x: (x == "unresolved").sum(),
            ),
            any_backward_score=("score_backward", "any"),
            any_multi_point_jump=("multi_point_jump", "any"),
            any_non_numeric_score=("score_is_numeric", lambda x: (~x).any()),
        )
        .reset_index()
        .merge(first_serve, on=group_cols, how="left")
        .merge(last_action, on=group_cols, how="left")
    )

    rally_table["away_delta"] = (
        rally_table["score_away_end"] - rally_table["score_away_start"]
    )
    rally_table["home_delta"] = (
        rally_table["score_home_end"] - rally_table["score_home_start"]
    )
    rally_table["point_delta_total"] = (
        rally_table["away_delta"].abs() + rally_table["home_delta"].abs()
    )
    rally_table["point_winner_side"] = np.select(
        # normal rally should move exactly one team by one point
        # if it does not, mark unknown instead of pretending
        [rally_table["home_delta"].eq(1), rally_table["away_delta"].eq(1)],
        ["home", "away"],
        default="unknown",
    )
    rally_table["point_winner_team"] = np.select(
        [
            rally_table["point_winner_side"].eq("home"),
            rally_table["point_winner_side"].eq("away"),
        ],
        [rally_table["home_clean"], rally_table["away_clean"]],
        default=pd.NA,
    )

    set_summary = (
        rally_table.sort_values(["contestid", "set", "rally_id"], kind="mergesort")
        .groupby(["contestid", "set"], sort=False, observed=True)
        .tail(1)[["contestid", "set", "score_home_end", "score_away_end"]]
        .copy()
    )
    set_summary["home_set_win"] = (
        set_summary["score_home_end"] > set_summary["score_away_end"]
    ).astype("int8")
    set_summary["away_set_win"] = (
        set_summary["score_away_end"] > set_summary["score_home_end"]
    ).astype("int8")
    set_summary = set_summary.sort_values(["contestid", "set"], kind="mergesort")
    home_cum_sets = set_summary.groupby("contestid", sort=False, observed=True)[
        "home_set_win"
    ].cumsum()
    away_cum_sets = set_summary.groupby("contestid", sort=False, observed=True)[
        "away_set_win"
    ].cumsum()
    set_summary["home_sets_before"] = (
        home_cum_sets - set_summary["home_set_win"]
    ).astype("int16")
    set_summary["away_sets_before"] = (
        away_cum_sets - set_summary["away_set_win"]
    ).astype("int16")
    contest_result = (
        set_summary.groupby("contestid", observed=True)
        .agg(
            home_sets_total=("home_set_win", "sum"),
            away_sets_total=("away_set_win", "sum"),
        )
        .reset_index()
    )
    contest_result["home_match_win"] = (
        contest_result["home_sets_total"] > contest_result["away_sets_total"]
    ).astype("int8")

    rally_table = rally_table.merge(
        set_summary[
            [
                "contestid",
                "set",
                "home_set_win",
                "away_set_win",
                "home_sets_before",
                "away_sets_before",
            ]
        ],
        on=["contestid", "set"],
        how="left",
    ).merge(contest_result, on="contestid", how="left")

    last_rally = rally_table.groupby(["contestid", "set"], observed=True)[
        "rally_id"
    ].transform("max")
    rally_table["is_set_terminal"] = rally_table["rally_id"].eq(last_rally)
    rally_table["home_sets_after"] = (
        rally_table["home_sets_before"]
        + (rally_table["is_set_terminal"] & rally_table["home_set_win"].eq(1))
    ).astype("int16")
    rally_table["away_sets_after"] = (
        rally_table["away_sets_before"]
        + (rally_table["is_set_terminal"] & rally_table["away_set_win"].eq(1))
    ).astype("int16")
    rally_table["set_after"] = (
        rally_table["set"].astype("int16")
        + rally_table["is_set_terminal"].astype("int16")
    ).astype("int16")
    rally_table["score_home_after_state"] = np.where(
        rally_table["is_set_terminal"], 0, rally_table["score_home_end"]
    )
    rally_table["score_away_after_state"] = np.where(
        rally_table["is_set_terminal"], 0, rally_table["score_away_end"]
    )
    rally_table["server_side"] = rally_table["server_side"].fillna("unknown")
    rally_table["server_side_after"] = rally_table["point_winner_side"].where(
        rally_table["point_winner_side"].isin(["home", "away"]), "unknown"
    )
    rally_table["season"] = season

    contest_master = (
        rally_table.groupby("contestid", observed=True)
        .agg(
            season=("season", "first"),
            date=("date", "first"),
            away_team=("away_team", "first"),
            home_team=("home_team", "first"),
            away_clean=("away_clean", "first"),
            home_clean=("home_clean", "first"),
            sets=("set", "nunique"),
            rallies=("rally_id", "size"),
            home_sets_total=("home_sets_total", "first"),
            away_sets_total=("away_sets_total", "first"),
            home_match_win=("home_match_win", "first"),
        )
        .reset_index()
    )

    event_cols = [
        "season",
        "date",
        "contestid",
        "set",
        "rally_id",
        "file_row",
        "row_in_rally",
        "action_seq",
        "away_clean",
        "home_clean",
        "away_score_ffill",
        "home_score_ffill",
        "team_clean",
        "actor_team",
        "actor_side",
        "actor_team_source",
        "event_family",
        "is_action_row",
        "player",
        "player_clean",
        "player_uid",
        "role_family",
    ]
    event_table = pbp[event_cols].copy()
    return contest_master, rally_table, event_table


def build_first_ball_tables(
    event_table: pd.DataFrame,
    rally_table: pd.DataFrame,
    player_master: pd.DataFrame,
    season: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # first ball stuff starts with the first serve recieve touch
    # reception errors count too because that is still an attempt
    group_cols = ["contestid", "set", "rally_id"]
    receive_event_families = ["reception", "reception_error"]
    first_reception = (
        event_table.loc[
            event_table["event_family"].isin(receive_event_families)
            & event_table["player_clean"].fillna("").ne("")
            & event_table["actor_team"].notna()
        ]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[
            group_cols + ["file_row", "actor_team", "player_clean", "event_family"]
        ]
        .rename(
            columns={
                "file_row": "reception_row",
                "actor_team": "receiving_team",
                "player_clean": "passer_player_clean",
                "event_family": "reception_event_family",
            }
        )
    )
    stage = event_table.merge(first_reception, on=group_cols, how="left")
    after_reception = stage["file_row"] > stage["reception_row"]
    same_team = stage["actor_team"].eq(stage["receiving_team"])
    opp_team = (
        stage["actor_team"].notna()
        & stage["receiving_team"].notna()
        & stage["actor_team"].ne(stage["receiving_team"])
    )
    attackish = stage["event_family"].isin(["attack", "kill", "attack_error"])
    core_action = stage["event_family"].isin(
        [
            "serve",
            "reception",
            "set",
            "attack",
            "kill",
            "ace",
            "dig",
            "block",
            "attack_error",
            "service_error",
            "reception_error",
            "set_error",
            "block_error",
            "dig_error",
        ]
    )

    first_attack = (
        # first same-team attack after the pass
        # this is how we know if the pass gave them a real first ball chance
        stage.loc[after_reception & same_team & attackish]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[
            group_cols + ["file_row", "player_clean", "event_family", "role_family"]
        ]
        .rename(
            columns={
                "file_row": "first_attack_row",
                "player_clean": "first_attack_player_clean",
                "event_family": "first_attack_event_family",
                "role_family": "first_attack_role_family",
            }
        )
    )
    first_opp_action = (
        stage.loc[after_reception & opp_team & core_action]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[group_cols + ["file_row"]]
        .rename(columns={"file_row": "first_opp_action_row"})
    )
    first_kill = (
        stage.loc[after_reception & same_team & stage["event_family"].eq("kill")]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[group_cols + ["file_row", "player_clean"]]
        .rename(
            columns={
                "file_row": "first_same_team_kill_row",
                "player_clean": "first_ball_kill_player_clean",
            }
        )
    )

    rally_pass = (
        rally_table[
            [
                "season",
                *group_cols,
                "date",
                "home_clean",
                "away_clean",
                "serve_team",
                "server_side",
                "server_player_clean",
                "terminal_event_family",
                "point_winner_side",
                "point_winner_team",
            ]
        ]
        .merge(first_reception, on=group_cols, how="left")
        .merge(first_attack, on=group_cols, how="left")
        .merge(first_opp_action, on=group_cols, how="left")
        .merge(first_kill, on=group_cols, how="left")
    )
    rally_pass["reception_error"] = rally_pass["reception_event_family"].eq(
        "reception_error"
    )
    rally_pass["receive_point_win"] = (
        rally_pass["point_winner_team"]
        .fillna("")
        .astype("string")
        .eq(rally_pass["receiving_team"].fillna("").astype("string"))
        & rally_pass["receiving_team"].notna()
    )
    rally_pass["first_ball_attack"] = (
        rally_pass["first_attack_row"].notna()
        & ~rally_pass["reception_error"]
        & (
            rally_pass["first_opp_action_row"].isna()
            | (rally_pass["first_attack_row"] < rally_pass["first_opp_action_row"])
        )
    )
    rally_pass["first_ball_kill"] = (
        # first ball kill = receiving team kills it before the other team
        # gets another real action
        rally_pass["first_same_team_kill_row"].notna()
        & rally_pass["first_ball_attack"]
        & ~rally_pass["reception_error"]
        & (
            rally_pass["first_opp_action_row"].isna()
            | (
                rally_pass["first_same_team_kill_row"]
                < rally_pass["first_opp_action_row"]
            )
        )
    )
    rally_pass["first_ball_middle_attack"] = (
        rally_pass["first_ball_attack"]
        & rally_pass["first_attack_role_family"].fillna("").eq("middle")
    )

    first_ball_player_contest = (
        rally_pass.loc[
            rally_pass["passer_player_clean"].notna()
            & rally_pass["receiving_team"].notna()
        ]
        .groupby(
            ["season", "contestid", "receiving_team", "passer_player_clean"],
            observed=True,
        )
        .agg(
            fb_receptions=("rally_id", "size"),
            fb_clean_receptions=("reception_error", lambda x: (~x).sum()),
            fb_reception_errors=("reception_error", "sum"),
            fb_attacks=("first_ball_attack", "sum"),
            fb_kills=("first_ball_kill", "sum"),
            fb_middle_attacks=("first_ball_middle_attack", "sum"),
            fb_receive_point_wins=("receive_point_win", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "receiving_team": "team_clean",
                "passer_player_clean": "player_clean",
            }
        )
    )
    for num, den, out in [
        ("fb_attacks", "fb_receptions", "fb_attack_rate"),
        ("fb_kills", "fb_receptions", "fb_kill_rate"),
        ("fb_middle_attacks", "fb_receptions", "fb_middle_attack_rate"),
        ("fb_reception_errors", "fb_receptions", "fb_reception_error_rate"),
        ("fb_receive_point_wins", "fb_receptions", "fb_receive_point_win_rate"),
    ]:
        first_ball_player_contest[out] = (
            first_ball_player_contest[num]
            / first_ball_player_contest[den].replace(0, np.nan)
        ).fillna(0.0)

    display_lookup = player_master[
        [
            "season",
            "team_clean",
            "player_clean",
            "player",
            "team",
            "role_family",
            "player_uid",
        ]
    ].drop_duplicates()
    first_ball_player_contest = first_ball_player_contest.merge(
        display_lookup, on=["season", "team_clean", "player_clean"], how="left"
    )
    first_ball_player_season = (
        first_ball_player_contest.groupby(
            ["season", "team_clean", "player_clean"], observed=True
        )
        .agg(
            fb_receptions=("fb_receptions", "sum"),
            fb_clean_receptions=("fb_clean_receptions", "sum"),
            fb_reception_errors=("fb_reception_errors", "sum"),
            fb_attacks=("fb_attacks", "sum"),
            fb_kills=("fb_kills", "sum"),
            fb_middle_attacks=("fb_middle_attacks", "sum"),
            fb_receive_point_wins=("fb_receive_point_wins", "sum"),
        )
        .reset_index()
    )
    first_ball_player_season["fb_attack_rate"] = (
        first_ball_player_season["fb_attacks"]
        / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season["fb_kill_rate"] = (
        first_ball_player_season["fb_kills"]
        / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season["fb_middle_attack_rate"] = (
        first_ball_player_season["fb_middle_attacks"]
        / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season["fb_reception_error_rate"] = (
        first_ball_player_season["fb_reception_errors"]
        / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season["fb_receive_point_win_rate"] = (
        first_ball_player_season["fb_receive_point_wins"]
        / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season = first_ball_player_season.merge(
        display_lookup, on=["season", "team_clean", "player_clean"], how="left"
    )
    first_ball_player_season["season"] = season
    return rally_pass, first_ball_player_contest, first_ball_player_season


def build_player_match_enriched(
    player_match: pd.DataFrame,
    team_strength: pd.DataFrame,
    first_ball_player_contest: pd.DataFrame,
) -> pd.DataFrame:
    # add team strength and first-ball stuff onto player match rows
    # this is the bridge from pbp back to the box score data
    out = player_match.copy()
    out = out.merge(
        team_strength[["season", "team_clean", "team_strength_index"]],
        on=["season", "team_clean"],
        how="left",
    )
    opp_strength = team_strength[
        ["season", "team_clean", "team_strength_index"]
    ].rename(
        columns={"team_clean": "opp_clean", "team_strength_index": "opp_strength_index"}
    )
    out = out.merge(opp_strength, on=["season", "opp_clean"], how="left")
    out["team_strength_missing"] = out["team_strength_index"].isna()
    out["opp_strength_missing"] = out["opp_strength_index"].isna()

    first_ball_small = first_ball_player_contest[
        [
            "season",
            "contestid",
            "team_clean",
            "player_clean",
            "fb_receptions",
            "fb_clean_receptions",
            "fb_reception_errors",
            "fb_attacks",
            "fb_kills",
            "fb_middle_attacks",
            "fb_receive_point_wins",
            "fb_attack_rate",
            "fb_kill_rate",
            "fb_middle_attack_rate",
            "fb_reception_error_rate",
            "fb_receive_point_win_rate",
        ]
    ].copy()
    first_ball_small["contestid"] = pd.to_numeric(
        first_ball_small["contestid"], errors="coerce"
    ).astype("Int64")
    out = out.merge(
        first_ball_small,
        on=["season", "contestid", "team_clean", "player_clean"],
        how="left",
    )

    for col in [
        "fb_receptions",
        "fb_clean_receptions",
        "fb_reception_errors",
        "fb_attacks",
        "fb_kills",
        "fb_middle_attacks",
        "fb_receive_point_wins",
    ]:
        out[col] = out[col].fillna(0).astype("int32")
    for col in [
        "fb_attack_rate",
        "fb_kill_rate",
        "fb_middle_attack_rate",
        "fb_reception_error_rate",
        "fb_receive_point_win_rate",
    ]:
        out[col] = out[col].fillna(0.0)
    for col in ["team_strength_index", "opp_strength_index"]:
        out[col] = out[col].fillna(0.0)
    return out


def build_player_season_features(
    player_master: pd.DataFrame,
    player_match_enriched: pd.DataFrame,
    first_ball_player_season: pd.DataFrame,
) -> pd.DataFrame:
    # roll match rows up to one row per player for the season
    # this is what model.py reads
    stat_cols = [
        "p",
        "s",
        "ms",
        "kills",
        "errors",
        "total_attacks",
        "assists",
        "aces",
        "serr",
        "digs",
        "rerr",
        "block_solos",
        "block_assists",
        "berr",
        "tb",
        "pts",
        "bhe",
        "retatt",
    ]
    pm = player_match_enriched.copy()
    for col in stat_cols:
        if col not in pm.columns:
            pm[col] = 0
        pm[col] = pd.to_numeric(pm[col], errors="coerce").fillna(0)
    pm["matches_played_rows"] = 1

    agg = (
        pm.groupby(["season", "team_clean", "player_clean"], observed=True)
        .agg(
            player=("player", "first"),
            team=("team", "first"),
            role_family=("role_family", "first"),
            player_uid=("player_uid", "first"),
            matches_played_rows=("matches_played_rows", "sum"),
            team_strength_index=("team_strength_index", "mean"),
            opp_strength_index=("opp_strength_index", "mean"),
            team_strength_missing_rate=("team_strength_missing", "mean"),
            opp_strength_missing_rate=("opp_strength_missing", "mean"),
            **{f"box_{col}": (col, "sum") for col in stat_cols},
        )
        .reset_index()
    )
    fb = first_ball_player_season[
        [
            "season",
            "team_clean",
            "player_clean",
            "fb_receptions",
            "fb_clean_receptions",
            "fb_reception_errors",
            "fb_attacks",
            "fb_kills",
            "fb_middle_attacks",
            "fb_receive_point_wins",
            "fb_attack_rate",
            "fb_kill_rate",
            "fb_middle_attack_rate",
            "fb_reception_error_rate",
            "fb_receive_point_win_rate",
        ]
    ].copy()
    features = player_master.merge(
        agg.drop(
            columns=["player", "team", "role_family", "player_uid"], errors="ignore"
        ),
        on=["season", "team_clean", "player_clean"],
        how="left",
    ).merge(fb, on=["season", "team_clean", "player_clean"], how="left")

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    for col in numeric_cols:
        features[col] = features[col].fillna(0)
    for col in [
        "fb_receptions",
        "fb_clean_receptions",
        "fb_reception_errors",
        "fb_attacks",
        "fb_kills",
        "fb_middle_attacks",
        "fb_receive_point_wins",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype("int32")
    for col in [
        "fb_attack_rate",
        "fb_kill_rate",
        "fb_middle_attack_rate",
        "fb_reception_error_rate",
        "fb_receive_point_win_rate",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cleaned volleyball data products."
    )
    add_season_args(parser)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--save-event-table", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_for_season(
    season: str,
    *,
    repo_root: Path | None = None,
    save_event_table: bool = True,
    download_missing: bool = True,
) -> None:
    package_root = repo_root or Path(Git(root_dir=".").root_dir)
    data_dir = package_root / "Data" / "original"
    out_dir = package_root / "Data" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root: {package_root}")
    print(f"season   : {season}")
    print(f"data_dir : {data_dir}")
    print(f"out_dir  : {out_dir}")

    players, player_match, team_match = load_season_source_files(
        data_dir,
        season,
        download_missing=download_missing,
    )
    player_master = build_player_master(players)
    player_match, team_match, canonical_team_df = prepare_match_tables(
        player_match, team_match, player_master
    )
    team_strength = build_team_strength(team_match, player_master)
    pbp = load_and_clean_pbp(
        data_dir,
        player_master,
        canonical_team_df["team_clean"].dropna().tolist(),
        season,
        download_missing=download_missing,
    )

    contest_master, rally_table, event_table = build_rally_tables(pbp, season)
    del pbp
    rally_pass, first_ball_player_contest, first_ball_player_season = (
        build_first_ball_tables(event_table, rally_table, player_master, season)
    )
    player_match_enriched = build_player_match_enriched(
        player_match, team_strength, first_ball_player_contest
    )
    player_season_features = build_player_season_features(
        player_master, player_match_enriched, first_ball_player_season
    )

    save_csv(player_master, out_dir, f"player_master_{season}")
    save_csv(team_strength, out_dir, f"team_strength_{season}")
    save_csv(contest_master, out_dir, f"contest_master_{season}")
    save_csv(rally_table, out_dir, f"rally_table_{season}")
    save_csv(rally_pass, out_dir, f"rally_pass_{season}")
    save_csv(
        first_ball_player_contest,
        out_dir,
        f"first_ball_pass_player_contest_{season}",
    )
    save_csv(
        first_ball_player_season,
        out_dir,
        f"first_ball_pass_player_season_{season}",
    )
    save_csv(player_match_enriched, out_dir, f"player_match_enriched_{season}")
    save_csv(player_season_features, out_dir, f"player_season_features_{season}")
    if save_event_table:
        save_csv(event_table, out_dir, f"event_table_{season}")


def main() -> None:
    # full rebuild every time
    # slower, but then old output files dont sneak into the project
    args = parse_args()
    package_root = Path(Git(root_dir=".").root_dir)
    data_dir = package_root / "Data" / "original"
    seasons = resolve_requested_seasons(
        season=args.season,
        seasons=args.seasons,
        season_range=args.season_range,
        all_seasons=args.all_seasons,
        data_dir=data_dir,
        download_missing=args.download_missing,
    )
    for season in seasons:
        run_for_season(
            season,
            repo_root=package_root,
            save_event_table=args.save_event_table,
            download_missing=args.download_missing,
        )


if __name__ == "__main__":
    main()
