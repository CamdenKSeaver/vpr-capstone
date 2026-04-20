# %%
import polars as pl
from scmrepo.git import Git
import os

PACKAGE_ROOT = Git(root_dir=".").root_dir

# make sure output folder exists
os.makedirs(f"{PACKAGE_ROOT}/Data/preprocessed", exist_ok=True)
os.makedirs(f"{PACKAGE_ROOT}/Data/cleaned", exist_ok=True)

# %%
player_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_player_match_2020_2025_master.csv"
)
team_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_team_match_2020_2025_master.csv"
)

print(player_match_df.head())
print(team_match_df.head())

# helper function to clean team names
def clean_team(col):
    return (
        col
        .str.replace_all(r"#\d+\s*", "")
        .str.replace_all(r"\(.*?\)", "")
        .str.replace_all(r"^@\s*", "")
        .str.replace_all(r"^vs\.?\s*", "")
        .str.replace_all(r"^at\s*", "")
        .str.split("@")
        .list.eval(pl.element().str.strip_chars())
        .list.filter(pl.element() != "")
        .list.get(0)
        .str.replace_all(r"\d{2}-\d{2}.*", "")
        .str.replace_all(r"\d{4}.*", "")
        .str.replace_all("-", " ")
        .str.strip_chars()
    )

# clean
player_df = player_match_df.with_columns([
    clean_team(pl.col("Team")).alias("Team"),
    clean_team(pl.col("Opponent Team")).alias("Opponent Team"),

    # convert location
    (pl.col("Location") == "Away").cast(pl.Int8).alias("is_away"),

    # efficiency
    pl.col("HitPct").alias("kill_efficiency"),

    # combine blocks
    (pl.col("BlockSolos") + pl.col("BlockAssists")).alias("Blocks"),
])

# fill nulls
player_df = player_df.with_columns([
    pl.col("Kills").fill_null(0),
    pl.col("Assists").fill_null(0),
    pl.col("Aces").fill_null(0),
    pl.col("Digs").fill_null(0),
    pl.col("BlockSolos").fill_null(0),
    pl.col("BlockAssists").fill_null(0),
    pl.col("Errors").fill_null(0),
    pl.col("TotalAttacks").fill_null(0),
])

# clean team match dataset
team_match_df = team_match_df.with_columns([
    clean_team(pl.col("Team")).alias("Team"),
    clean_team(pl.col("Opponent")).alias("Opponent"),
])

# get rid of any postponed or canceled matches
team_match_df = team_match_df.filter(
    ~pl.col("Result").is_in(["Ppd", "Canceled"])
)

# player
player_stats = player_df.select([
    "Player",
    "Team",
    "Opponent Team",
    "Season",
    "is_away",
    "Kills",
    "Assists",
    "Aces",
    "Digs",
    "Blocks",
    pl.col("kill_efficiency").round(4)
])

# rename column 
player_stats = player_stats.rename({
    "Opponent Team": "Opponent"
})

# sort player data
player_stats = player_stats.sort(["Team", "Player", "Season"])

# team
team_stats = player_df.group_by(["Team", "Season"]).agg([
    pl.mean("Kills").round(4).alias("avg_kills"),
    pl.mean("Assists").round(4).alias("avg_assists"),
    pl.mean("Digs").round(4).alias("avg_digs"),
    pl.mean("Blocks").round(4).alias("avg_blocks"),

    # correct team efficiency
    (
        (pl.sum("Kills") - pl.sum("Errors")) /
        pl.sum("TotalAttacks")
    ).round(4).alias("avg_efficiency"),

    pl.len().alias("matches_played")
])

# sort team data
team_stats = team_stats.sort(["Team", "Season"])

# match
match_stats = team_match_df.select([
    "Team",
    "Opponent",
    "Season",
    "Result"
])

# save output
player_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/player_stats.csv")
team_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/team_stats.csv")
match_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/match_stats.csv")

# clean original files
player_df.write_csv(f"{PACKAGE_ROOT}/Data/cleaned/player_match_cleaned.csv")
team_match_df.write_csv(f"{PACKAGE_ROOT}/Data/cleaned/team_match_cleaned.csv")

# %%
kills_df = player_df.select(
    "Player",
    "Team",
    "Opponent Team",
    "Season",
    "Kills"
)

kills_df = kills_df.rename({
    "Opponent Team": "Opponent"
})

kills_df.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/kills.csv")

# %%