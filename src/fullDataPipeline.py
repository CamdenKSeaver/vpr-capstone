"""Full data pipeline for the NCAA womens volleyball project.

Takes the raw csvs and makes the cleaned csvs that model.py uses.
This is kind of a big file, but it is nice for the group because there is
one obvious thing to run when the data needs to be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEASON = os.environ.get("VPR_SEASON", "2025-2026")

# only keeping the pbp columns I actually use
# the raw file has a bunch more but it just makes this take forever
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
    # position labels are kinda all over the place
    # this gets them into the buckets we rank on later
    # there are definetly weird DS/OH type players, model.py checks again too
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
    # random team name stuff I had to hand fix while checking joins
    # if a team is missing later this is probaly where I would look first
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


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Data" / "original").exists():
            return candidate
    return here


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # make columns not annoying before doing joins
    # some files say hitpct and some say hit_pct etc
    rename_map = {
        c: re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip("_")
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
        # this can happen after renaming columns
        # I am assuming the left most non empty value is the safest
        same = df.loc[:, df.columns == name]
        merged = same.bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=name)
        df[name] = merged
    return df


def clean_series(s: pd.Series) -> pd.Series:
    # clean names pretty hard so roster/match/pbp will actually join
    # it does lose accents which is not ideal but the joins work better
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
    # opponent fields have @, rankings, tourney names, notes, etc
    # good for people reading it, bad for matching teams
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
    # try the hand fixes first, then exact names, then the compact version
    # I would rather leave something messy than match to the wrong team
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


def load_csv_flexible(path_or_url: Path | str, desired_cols=None, dtype_map=None) -> pd.DataFrame:
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


def canonical_event_family(event_text: pd.Series, description_text: pd.Series) -> pd.Series:
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
        ("set_error", r"set error|ball handling error", r"set error|ball handling error"),
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

    admin_mask = (
        evt.str.contains(r"timeout|challenge|sanction|sub|substitution|media timeout|\+", regex=True, na=False)
        | desc.str.contains(r"timeout|challenge|sanction|sub|substitution|media timeout", regex=True, na=False)
    )
    family = family.mask(family.isna() & admin_mask, "admin")
    return family.fillna("admin")


def load_aggregate_files(data_dir: Path, season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # load the normal roster / box score files first
    # pbp is huge so I deal with it later
    players = coalesce_duplicate_columns(
        standardize_columns(pd.read_csv(data_dir / "all_players.csv", low_memory=False))
    )
    player_match = coalesce_duplicate_columns(
        standardize_columns(
            pd.read_csv(data_dir / "women_d1_player_match_2020_2025_master.csv", low_memory=False)
        )
    )
    team_match = coalesce_duplicate_columns(
        standardize_columns(
            pd.read_csv(data_dir / "women_d1_team_match_2020_2025_master.csv", low_memory=False)
        )
    )

    players = players.loc[players["season"].astype("string").eq(season)].copy()
    player_match = player_match.loc[player_match["season"].astype("string").eq(season)].copy()
    team_match = team_match.loc[team_match["season"].astype("string").eq(season)].copy()

    if "player" in players.columns:
        bad_player_labels = {"team", "totals", "opponent totals", "opponent"}
        players = players.loc[~clean_series(players["player"]).isin(bad_player_labels)].copy()

    player_match = player_match.drop_duplicates().copy()
    return players, player_match, team_match


def build_player_master(players: pd.DataFrame) -> pd.DataFrame:
    # main player identity table
    # most later joins are basically team_clean + player_clean
    players = players.copy()
    players["player_clean"] = clean_series(players["player"])
    players["team_clean"] = clean_team_series(players["team"])
    players["pos_clean"] = clean_series(players["pos"]) if "pos" in players.columns else ""
    players["role_family"] = players["pos_clean"].map(ROLE_MAP).fillna("unknown")
    players["number_clean"] = pd.to_numeric(players.get("number"), errors="coerce").astype("Int64")
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
        player_match["contestid"] = pd.to_numeric(player_match["contestid"], errors="coerce").astype("Int64")
    else:
        player_match["contestid"] = pd.Series(pd.NA, index=player_match.index, dtype="Int64")

    return player_match, team_match, pd.DataFrame({"team_clean": canonical_teams})


def build_team_strength(team_match: pd.DataFrame, player_master: pd.DataFrame) -> pd.DataFrame:
    # simple team strength, not some full elo thing
    # just enough to control for good/bad teams later
    team_match = team_match.copy()
    for col in ["hit_pct", "s", "kills", "errors", "assists", "aces", "serr", "digs", "pts"]:
        if col not in team_match.columns:
            team_match[col] = np.nan

    result_parts = team_match["result"].fillna("").astype("string").str.extract(
        r"^\s*([WL])\s*(\d+)\s*-\s*(\d+)\s*$"
    )
    team_match["match_win"] = result_parts[0].fillna("").eq("W").astype("int8")
    team_match["sets_for"] = pd.to_numeric(result_parts[1], errors="coerce").fillna(0).astype("int16")
    team_match["sets_against"] = pd.to_numeric(result_parts[2], errors="coerce").fillna(0).astype("int16")
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
        + 0.15 * zscore(team_strength["mean_hit_pct"].fillna(team_strength["mean_hit_pct"].median()))
    )
    team_names = (
        player_master[["team_clean", "team"]]
        .drop_duplicates("team_clean")
        .rename(columns={"team": "team_display"})
    )
    return team_strength.merge(team_names, on="team_clean", how="left")


def load_and_clean_pbp(
    data_dir: Path,
    player_master: pd.DataFrame,
    canonical_teams: list[str],
    season: str,
) -> tuple[pd.DataFrame, dict[str, int | float | str | bool]]:
    # this is the slow part
    # pbp is big, so dont do random stuff here or it gets hard to debug
    pbp_path = data_dir / "wvb_pbp_div1_2025.csv"
    if not pbp_path.exists():
        raise FileNotFoundError(f"Missing play-by-play file: {pbp_path}")

    print(f"loading play-by-play: {pbp_path}")
    pbp_raw = load_csv_flexible(pbp_path, PBP_DESIRED_COLS, PBP_DTYPE_MAP)
    raw_rows = len(pbp_raw)
    pbp = pbp_raw.drop_duplicates().copy()
    duplicate_exact_rows = raw_rows - len(pbp)
    del pbp_raw

    pbp = standardize_columns(pbp)
    pbp["season"] = season
    pbp["file_row"] = np.arange(len(pbp), dtype="int64")
    pbp["contestid"] = pd.to_numeric(pbp["contestid"], errors="coerce").astype("Int64")
    pbp["set"] = pd.to_numeric(pbp["set"], errors="coerce").astype("Int16")
    pbp = pbp.dropna(subset=["contestid", "set"]).copy()

    for col in ["team", "home_team", "away_team", "player", "event", "description", "score", "date"]:
        if col not in pbp.columns:
            pbp[col] = pd.Series("", index=pbp.index, dtype="string")

    pbp["team_clean"] = canonicalize_vs_known_teams(pbp["team"], canonical_teams, MANUAL_TEAM_ALIASES)
    pbp["home_clean"] = canonicalize_vs_known_teams(pbp["home_team"], canonical_teams, MANUAL_TEAM_ALIASES)
    pbp["away_clean"] = canonicalize_vs_known_teams(pbp["away_team"], canonical_teams, MANUAL_TEAM_ALIASES)
    pbp["player_clean"] = clean_series(pbp["player"])
    pbp["event_text"] = clean_series(pbp["event"])
    pbp["description_text"] = clean_series(pbp["description"])

    sort_cols = ["contestid", "set", "file_row"]
    pbp = pbp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    roster_key = player_master[["player_clean", "team_clean"]].drop_duplicates()
    # figure out who the actor actually plays for
    # raw team column gets weird on errors so roster lookup gets first shot
    home_lookup = roster_key.rename(columns={"team_clean": "home_clean"}).assign(player_on_home=True)
    away_lookup = roster_key.rename(columns={"team_clean": "away_clean"}).assign(player_on_away=True)
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
        [pbp["actor_team"].eq(pbp["home_clean"]), pbp["actor_team"].eq(pbp["away_clean"])],
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

    score_parts = pbp["score"].astype("string").str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    # scores are how I rebuild rallies
    # weird score rows stay in, but they get flags so we can see them later
    pbp["away_score_raw"] = pd.to_numeric(score_parts[0], errors="coerce")
    pbp["home_score_raw"] = pd.to_numeric(score_parts[1], errors="coerce")
    pbp["score_is_numeric"] = pbp["away_score_raw"].notna() & pbp["home_score_raw"].notna()
    group_set = pbp.groupby(["contestid", "set"], sort=False, observed=True)
    pbp[["away_score_ffill", "home_score_ffill"]] = group_set[["away_score_raw", "home_score_raw"]].ffill()
    pbp[["away_score_ffill", "home_score_ffill"]] = pbp[
        ["away_score_ffill", "home_score_ffill"]
    ].fillna(0)
    pbp["d_away"] = group_set["away_score_ffill"].diff().fillna(0)
    pbp["d_home"] = group_set["home_score_ffill"].diff().fillna(0)
    pbp["score_backward"] = (pbp["d_away"] < 0) | (pbp["d_home"] < 0)
    pbp["multi_point_jump"] = (pbp["d_away"].abs() + pbp["d_home"].abs()) > 1
    pbp["one_point_increment"] = (
        (pbp["d_away"] >= 0) & (pbp["d_home"] >= 0) & ((pbp["d_away"] + pbp["d_home"]) == 1)
    )

    pbp["event_family"] = canonical_event_family(pbp["event_text"], pbp["description_text"])
    pbp["is_first_ball_kill_text"] = (
        pbp["event_text"].str.contains(r"first ball kill", na=False)
        | pbp["description_text"].str.contains(r"first ball kill", na=False)
    )
    pbp["is_action_row"] = pbp["event_family"].ne("admin")

    prev_end = (
        pbp.groupby(["contestid", "set"], sort=False, observed=True)["one_point_increment"]
        .shift(fill_value=False)
    )
    pbp["rally_id"] = (
        prev_end.groupby([pbp["contestid"], pbp["set"]], sort=False)
        .cumsum()
        .astype("int32")
        + 1
    )
    group_rally = ["contestid", "set", "rally_id"]
    pbp["row_in_rally"] = (
        pbp.groupby(group_rally, sort=False, observed=True).cumcount().astype("int16") + 1
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

    stats = {
        "raw_pbp_rows": int(raw_rows),
        "deduped_pbp_rows": int(len(pbp)),
        "duplicate_exact_pbp_rows": int(duplicate_exact_rows),
        "duplicate_exact_pbp_rate": float(duplicate_exact_rows / raw_rows) if raw_rows else 0.0,
        "non_numeric_score_rows": int((~pbp["score_is_numeric"]).sum()),
        "backward_score_rows": int(pbp["score_backward"].sum()),
        "multi_point_jump_rows": int(pbp["multi_point_jump"].sum()),
        "unresolved_actor_rows": int((pbp["actor_team_source"] == "unresolved").sum()),
    }
    return pbp, stats


def build_rally_tables(pbp: pd.DataFrame, season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        .reset_index()[group_cols + ["file_row", "actor_team", "actor_side", "player_clean"]]
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
        .tail(1)[group_cols + ["file_row", "event_family", "actor_team", "actor_side", "player_clean"]]
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
            unresolved_actor_rows=("actor_team_source", lambda x: (x == "unresolved").sum()),
            any_backward_score=("score_backward", "any"),
            any_multi_point_jump=("multi_point_jump", "any"),
            any_non_numeric_score=("score_is_numeric", lambda x: (~x).any()),
        )
        .reset_index()
        .merge(first_serve, on=group_cols, how="left")
        .merge(last_action, on=group_cols, how="left")
    )

    rally_table["away_delta"] = rally_table["score_away_end"] - rally_table["score_away_start"]
    rally_table["home_delta"] = rally_table["score_home_end"] - rally_table["score_home_start"]
    rally_table["point_delta_total"] = rally_table["away_delta"].abs() + rally_table["home_delta"].abs()
    rally_table["point_winner_side"] = np.select(
        # normal rally should move exactly one team by one point
        # if it does not, mark unknown instead of pretending
        [rally_table["home_delta"].eq(1), rally_table["away_delta"].eq(1)],
        ["home", "away"],
        default="unknown",
    )
    rally_table["point_winner_team"] = np.select(
        [rally_table["point_winner_side"].eq("home"), rally_table["point_winner_side"].eq("away")],
        [rally_table["home_clean"], rally_table["away_clean"]],
        default=pd.NA,
    )

    set_summary = (
        rally_table.sort_values(["contestid", "set", "rally_id"], kind="mergesort")
        .groupby(["contestid", "set"], sort=False, observed=True)
        .tail(1)[["contestid", "set", "score_home_end", "score_away_end"]]
        .copy()
    )
    set_summary["home_set_win"] = (set_summary["score_home_end"] > set_summary["score_away_end"]).astype("int8")
    set_summary["away_set_win"] = (set_summary["score_away_end"] > set_summary["score_home_end"]).astype("int8")
    set_summary = set_summary.sort_values(["contestid", "set"], kind="mergesort")
    home_cum_sets = set_summary.groupby("contestid", sort=False, observed=True)["home_set_win"].cumsum()
    away_cum_sets = set_summary.groupby("contestid", sort=False, observed=True)["away_set_win"].cumsum()
    set_summary["home_sets_before"] = (home_cum_sets - set_summary["home_set_win"]).astype("int16")
    set_summary["away_sets_before"] = (away_cum_sets - set_summary["away_set_win"]).astype("int16")
    contest_result = (
        set_summary.groupby("contestid", observed=True)
        .agg(home_sets_total=("home_set_win", "sum"), away_sets_total=("away_set_win", "sum"))
        .reset_index()
    )
    contest_result["home_match_win"] = (
        contest_result["home_sets_total"] > contest_result["away_sets_total"]
    ).astype("int8")

    rally_table = rally_table.merge(
        set_summary[["contestid", "set", "home_set_win", "away_set_win", "home_sets_before", "away_sets_before"]],
        on=["contestid", "set"],
        how="left",
    ).merge(contest_result, on="contestid", how="left")

    last_rally = rally_table.groupby(["contestid", "set"], observed=True)["rally_id"].transform("max")
    rally_table["is_set_terminal"] = rally_table["rally_id"].eq(last_rally)
    rally_table["home_sets_after"] = (
        rally_table["home_sets_before"] + (rally_table["is_set_terminal"] & rally_table["home_set_win"].eq(1))
    ).astype("int16")
    rally_table["away_sets_after"] = (
        rally_table["away_sets_before"] + (rally_table["is_set_terminal"] & rally_table["away_set_win"].eq(1))
    ).astype("int16")
    rally_table["set_after"] = (
        rally_table["set"].astype("int16") + rally_table["is_set_terminal"].astype("int16")
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
        .reset_index()[group_cols + ["file_row", "actor_team", "player_clean", "event_family"]]
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
    opp_team = stage["actor_team"].notna() & stage["receiving_team"].notna() & stage["actor_team"].ne(
        stage["receiving_team"]
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
        .reset_index()[group_cols + ["file_row", "player_clean", "event_family"]]
        .rename(
            columns={
                "file_row": "first_attack_row",
                "player_clean": "first_attack_player_clean",
                "event_family": "first_attack_event_family",
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
    rally_pass["reception_error"] = rally_pass["reception_event_family"].eq("reception_error")
    rally_pass["receive_point_win"] = rally_pass["point_winner_team"].fillna("").astype("string").eq(
        rally_pass["receiving_team"].fillna("").astype("string")
    ) & rally_pass["receiving_team"].notna()
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
            | (rally_pass["first_same_team_kill_row"] < rally_pass["first_opp_action_row"])
        )
    )

    first_ball_player_contest = (
        rally_pass.loc[rally_pass["passer_player_clean"].notna() & rally_pass["receiving_team"].notna()]
        .groupby(["season", "contestid", "receiving_team", "passer_player_clean"], observed=True)
        .agg(
            fb_receptions=("rally_id", "size"),
            fb_clean_receptions=("reception_error", lambda x: (~x).sum()),
            fb_reception_errors=("reception_error", "sum"),
            fb_attacks=("first_ball_attack", "sum"),
            fb_kills=("first_ball_kill", "sum"),
            fb_receive_point_wins=("receive_point_win", "sum"),
        )
        .reset_index()
        .rename(columns={"receiving_team": "team_clean", "passer_player_clean": "player_clean"})
    )
    for num, den, out in [
        ("fb_attacks", "fb_receptions", "fb_attack_rate"),
        ("fb_kills", "fb_receptions", "fb_kill_rate"),
        ("fb_reception_errors", "fb_receptions", "fb_reception_error_rate"),
        ("fb_receive_point_wins", "fb_receptions", "fb_receive_point_win_rate"),
    ]:
        first_ball_player_contest[out] = (
            first_ball_player_contest[num] / first_ball_player_contest[den].replace(0, np.nan)
        ).fillna(0.0)

    display_lookup = player_master[
        ["season", "team_clean", "player_clean", "player", "team", "role_family", "player_uid"]
    ].drop_duplicates()
    first_ball_player_contest = first_ball_player_contest.merge(
        display_lookup, on=["season", "team_clean", "player_clean"], how="left"
    )
    first_ball_player_season = (
        first_ball_player_contest.groupby(["season", "team_clean", "player_clean"], observed=True)
        .agg(
            fb_receptions=("fb_receptions", "sum"),
            fb_clean_receptions=("fb_clean_receptions", "sum"),
            fb_reception_errors=("fb_reception_errors", "sum"),
            fb_attacks=("fb_attacks", "sum"),
            fb_kills=("fb_kills", "sum"),
            fb_receive_point_wins=("fb_receive_point_wins", "sum"),
        )
        .reset_index()
    )
    first_ball_player_season["fb_attack_rate"] = (
        first_ball_player_season["fb_attacks"] / first_ball_player_season["fb_receptions"].replace(0, np.nan)
    ).fillna(0.0)
    first_ball_player_season["fb_kill_rate"] = (
        first_ball_player_season["fb_kills"] / first_ball_player_season["fb_receptions"].replace(0, np.nan)
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
    opp_strength = team_strength[["season", "team_clean", "team_strength_index"]].rename(
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
            "fb_receive_point_wins",
            "fb_attack_rate",
            "fb_kill_rate",
            "fb_reception_error_rate",
            "fb_receive_point_win_rate",
        ]
    ].copy()
    first_ball_small["contestid"] = pd.to_numeric(first_ball_small["contestid"], errors="coerce").astype("Int64")
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
        "fb_receive_point_wins",
    ]:
        out[col] = out[col].fillna(0).astype("int32")
    for col in [
        "fb_attack_rate",
        "fb_kill_rate",
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
            "fb_receive_point_wins",
            "fb_attack_rate",
            "fb_kill_rate",
            "fb_reception_error_rate",
            "fb_receive_point_win_rate",
        ]
    ].copy()
    features = player_master.merge(
        agg.drop(columns=["player", "team", "role_family", "player_uid"], errors="ignore"),
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
        "fb_receive_point_wins",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype("int32")
    for col in [
        "fb_attack_rate",
        "fb_kill_rate",
        "fb_reception_error_rate",
        "fb_receive_point_win_rate",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned volleyball data products.")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--save-event-table", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    # full rebuild every time
    # slower, but then old output files dont sneak into the project
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    data_dir = repo_root / "Data" / "original"
    out_dir = repo_root / "Data" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root: {repo_root}")
    print(f"season   : {args.season}")
    print(f"data_dir : {data_dir}")
    print(f"out_dir  : {out_dir}")

    players, player_match, team_match = load_aggregate_files(data_dir, args.season)
    player_master = build_player_master(players)
    player_match, team_match, canonical_team_df = prepare_match_tables(
        player_match, team_match, player_master
    )
    team_strength = build_team_strength(team_match, player_master)
    pbp, pbp_stats = load_and_clean_pbp(
        data_dir,
        player_master,
        canonical_team_df["team_clean"].dropna().tolist(),
        args.season,
    )

    contest_master, rally_table, event_table = build_rally_tables(pbp, args.season)
    del pbp
    rally_pass, first_ball_player_contest, first_ball_player_season = build_first_ball_tables(
        event_table, rally_table, player_master, args.season
    )
    player_match_enriched = build_player_match_enriched(
        player_match, team_strength, first_ball_player_contest
    )
    player_season_features = build_player_season_features(
        player_master, player_match_enriched, first_ball_player_season
    )

    save_csv(player_master, out_dir, f"player_master_{args.season}")
    save_csv(team_strength, out_dir, f"team_strength_{args.season}")
    save_csv(contest_master, out_dir, f"contest_master_{args.season}")
    save_csv(rally_table, out_dir, f"rally_table_{args.season}")
    save_csv(rally_pass, out_dir, f"rally_pass_{args.season}")
    save_csv(first_ball_player_contest, out_dir, f"first_ball_pass_player_contest_{args.season}")
    save_csv(first_ball_player_season, out_dir, f"first_ball_pass_player_season_{args.season}")
    save_csv(player_match_enriched, out_dir, f"player_match_enriched_{args.season}")
    save_csv(player_season_features, out_dir, f"player_season_features_{args.season}")
    if args.save_event_table:
        save_csv(event_table, out_dir, f"event_table_{args.season}")

    manifest = {
        "season": args.season,
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "players_rows": int(len(players)),
        "player_match_rows": int(len(player_match)),
        "team_match_rows": int(len(team_match)),
        "player_master_rows": int(len(player_master)),
        "team_strength_rows": int(len(team_strength)),
        "contest_master_rows": int(len(contest_master)),
        "rally_rows": int(len(rally_table)),
        "event_rows": int(len(event_table)),
        "first_ball_player_contest_rows": int(len(first_ball_player_contest)),
        "first_ball_player_season_rows": int(len(first_ball_player_season)),
        "player_match_enriched_rows": int(len(player_match_enriched)),
        "player_season_features_rows": int(len(player_season_features)),
        **pbp_stats,
    }
    manifest_path = out_dir / f"preprocess_manifest_{args.season}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
