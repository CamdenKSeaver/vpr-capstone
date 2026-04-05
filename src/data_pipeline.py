# %%
import polars as pl
from scmrepo.git import Git

PACKAGE_ROOT = Git(root_dir=".").root_dir

# %%
player_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_player_match_2020_2025_master.csv"
)
team_match_df = pl.read_csv(
    f"{PACKAGE_ROOT}/Data/original/women_d1_team_match_2020_2025_master.csv"
)
print(player_match_df.head())
print(team_match_df.head())

# %%
kills_df = player_match_df.select(
    pl.col("Team"), pl.col("Opponent Team"), pl.col("Location"), pl.col("Kills")
)

# %%
kills_df.write_csv(f"{PACKAGE_ROOT}/Data/preprocessed/kills.csv")

# %%
