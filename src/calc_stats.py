import pandas as pd

"date","set","away_team","home_team","score","team","event","player","description"

"Season","Team","Conference","Number","Player","Yr","Pos","Ht","GP","GS","S","MS","Kills","Errors","Total Attacks","Hit Pct","Assists","Aces","SErr","Digs","RErr","Block Solos","Block Assists","BErr","TB","PTS","BHE","Trpl Dbl"
def calc_stats(match_data : pd.DataFrame, player_data : pd.DataFrame):
    for keys, game in match_data.groupby(["date", "set", "away_team", "home_team"]):
        team = game.iloc[0]["team"]

        away_team_players = game.loc[game["team"] == game.away_team, "player"].unique()
        home_team_players = game.loc[game["team"] == game.home_team, "player"].unique()
        for play in game.itertuples(index=False):
            if play.team == play.away_team:
                player_data.loc["player" in away_team_players][f"{play.event}_opportunity"] += 1
                player_data.loc["player"==play.player][f"{play.event}s"] += 1
            else:
                player_data.loc["player" in home_team_players][f"{play.event}_opportunity"] += 1
                player_data.loc["player"==play.player][f"{play.event}s"] += 1