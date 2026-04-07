# %%
import polars as pl
from scmrepo.git import Git
import os

PACKAGE_ROOT = Git(root_dir=".").root_dir

os.makedirs(f"{PACKAGE_ROOT}/Data/preprocessed", exist_ok=True)

# %%
player_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_player_match_2020_2025_master.csv"
)
team_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_team_match_2020_2025_master.csv"
)

print(player_match_df.head())
print(team_match_df.head())

# clean
player_df = player_match_df.with_columns([
    # Convert location to numeric
    (pl.col("Location") == "Away").cast(pl.Int8).alias("is_away"),

    # Use existing efficiency metric
    pl.col("HitPct").alias("kill_efficiency"),

    # Combine blocks
    (pl.col("BlockSolos") + pl.col("BlockAssists")).alias("Blocks")
])

# player
player_stats = player_df.select([
    "Player",
    "Team",
    "Opponent Team",
    "is_away",
    "Kills",
    "Assists",
    "Aces",
    "Digs",
    "Blocks",
    "kill_efficiency"
])

# team
team_stats = player_df.group_by(["Team", "Season"]).agg([
    pl.mean("Kills").round(4).alias("avg_kills"),
    pl.mean("Assists").round(4).alias("avg_assists"),
    pl.mean("Digs").round(4).alias("avg_digs"),
    pl.mean("Blocks").round(4).alias("avg_blocks"),

    # efficiency
    (
        (pl.sum("Kills") - pl.sum("Errors")) /
        pl.sum("TotalAttacks")
    ).round(4).alias("avg_efficiency"),

    pl.len().alias("matches_played")

])
# sort team stats
team_stats = team_stats.sort(["Team", "Season"])

# match
match_stats = team_match_df.select([
    "Team",
    "Opponent",
    "Result"
])

# save output
player_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/player_stats.csv")
team_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/team_stats.csv")
match_stats.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/match_stats.csv")

# %%
kills_df = player_match_df.select(
    "Player", "Team", "Opponent Team", "Location", "Kills"
)

# %%
kills_df.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/kills.csv")

# %%