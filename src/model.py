# %%
import polars as pl
from scmrepo.git import Git
import statsmodels.api as sm

PACKAGE_ROOT = Git(root_dir=".").root_dir
# %%
kills_df = pl.read_csv(f"{PACKAGE_ROOT}/Data/preprocessed/kills.csv")
kills_df = kills_df.with_columns(
    (pl.col("Location") == "Away").cast(pl.Int8).alias("Location")
)  # Boolean - Away = 1, Home = 0

# %%
x = kills_df.select("Location").to_series().to_numpy()
y = kills_df.select("Kills").to_series().to_numpy()

model = sm.OLS(y, sm.add_constant(x)).fit()

print(model.summary())

"""
The simple model implies that each player gets an average of 4.23 kills in home games, and 4.03 kills in away games.
"""

# %%
# Output a small csv file with the model predictions for locations 0 and 1 (home and away)
pred_df = pl.DataFrame({"Location": [0, 1]})

x_pred = sm.add_constant(pred_df.get_column("Location").to_numpy())
pred_kills = model.predict(x_pred)

pred_df = pred_df.with_columns(pl.Series("pred_kills", pred_kills))
print(pred_df)
pred_df.write_csv(f"{PACKAGE_ROOT}/Data/model_outputs/kills_by_location.csv")


# %%
