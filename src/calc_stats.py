import pandas as pd
from game import Game


#This method calculates the relevant match stats for the ranking system
#Currently it does not do anything relevant with them
#We probably want to store a csv for each game
def calc_stats(match_data : pd.DataFrame, player_data : pd.DataFrame):
    #This loop runs once for each game in a match
    for keys, game_data in match_data.groupby(["date", "set", "away_team", "home_team"]):
        #Finding the players on each side of the game - this information is not easily available in the data
        #This works by finding all the players who have an event attributed to them in the play by play
        #There are multiple things that need to be fixed with this approach : 
        #Some events like blocks have multiple players
        #Error events should not be included in this as the "team" column is not sufficient to interpret what team they play for
        #The data does not contain a way to know what players are on the field
            #As compensation we divide contributions to the player data by the number of players on the team
        away_team_players = game_data.loc[game_data["team"] == game_data.away_team, "player"].unique()
        home_team_players = game_data.loc[game_data["team"] == game_data.home_team, "player"].unique()

        game = Game(game_data, away_team_players, home_team_players)

        #Keep track of relevant action each serve
        actions = {}

        #This is only here to generate test data
        for player in away_team_players:
            row = {col: 0.0 for col in player_data.columns}
            row["player"] = player

            player_data.loc[len(player_data)] = row

        for player in home_team_players:
            row = {col: 0.0 for col in player_data.columns}
            row["player"] = player

            player_data.loc[len(player_data)] = row

        #Step by play-by-play
        for play in game_data.itertuples(index=False):
            #This block does not work for serves
            #This if statement is responsible for generating indivudal player stats that are not recorded in the availalbe data
            #Anytime a player in named in an event it records that event in their personal record of all their events and in the game record
            #Additionally all players on their team (including themself) are marked as having an opportunity for that event
            #This allows us to calculate what the percent chance that a player will do an event if given the chance - better players should have higher percents
            #We only add a fraction of an opportunity each time as we don't have a way of knowing who is on the field at any time
            #So we average out the opportunity

            #This section does not make proper use of the wrapper classes and I will rework it to do so
            if play.team == play.away_team:
                for player in away_team_players:
                    game.player_data.loc[player_data["player"] == player, f"{play.event}_opportunity"] += 1
                    player_data.loc[player_data["player"] == player, f"{play.event}s"] += 1 / len(away_team_players)

                player_data.loc[player_data["player"]==play.player, f"{play.event}s"] += 1 / len(away_team_players)
                game.player_data.loc[player_data["player"]==play.player, f"{play.event}s"] += 1
            else:
                for player in away_team_players:
                    game.player_data.loc[player_data["player"] == player, f"{play.event}_opportunity"] += 1
                    player_data.loc[player_data["player"] == player, f"{play.event}s"] += 1 / len(home_team_players)
                
                player_data.loc[player_data["player"]==play.player, f"{play.event}s"] += 1 / len(home_team_players)
                game.player_data.loc[player_data["player"]==play.player, f"{play.event}s"] += 1


            #This section is responsible for calculating team stats in a game
            #Each serve has a dictionary called actions associated with it
            #We use that dictionary to tally up occurrences of events and occurences of events given other ones have alreay happened
            #This is how we populate the functions of the form avg_team_sets_per_opp_return in game.py
            #I have a couple of these down but still have 7 left to go so feel free to work on them as well
            match play.event:
                case "Serve":
                    if len(actions.keys()) > 0:
                        serve_team = actions["Serve_team"]
                        receive_team = actions["Receive_team"]
                        game.same_team_sets_from_serve[serve_team] += actions[f"{serve_team}_sets"]

                        #Here we are calculating the average number of attacks the opposing team will perform after receiving a serve. 
                        #This is one of the more simple ones where there can only be one serve per serve so we just add the total number of attacks by the receiving team
                        game.opp_team_attacks_from_serve[receive_team] += actions[f"{receive_team}_attacks"]
                    actions = {
                        "Serve_team" : play.team,
                        "Receive_team" : "",
                        f"{play.away_team}_sets" : 0,
                        f"{play.home_team}_sets" : 0,
                        f"{play.away_team}_digs" : 0,
                        f"{play.home_team}_digs" : 0,
                        f"{play.away_team}_attacks" : 0,
                        f"{play.home_team}_attacks" : 0
                    }
                    if play.team == play.away_team:
                        actions["Receive_team"] = play.home_team
                    else:
                        actions["Receive_team"] = play.away_team
                case "Service error":
                    pass

                case "Reception":
                    pass

                case "Set":
                    #Here in set we might need to calculate something like average number of sets by the same team after a dig
                    #To do this every time there is a set we add the number of previous digs to the relevant place
                    actions[f"{play.team}_sets"] += 1
                    #game.same_team_sets_after_digs[play.team] += actions[f"{play.team}_digs"]

                case "Attack":
                    actions[f"{play.team}_sets"] += 1
        #print(game.player_data)
        print(player_data)

if __name__ == "__main__":
    players = pd.DataFrame(columns=[
        "player", 
        "Team", 
        "id",
        "Serves", 
        "Serve_opportunity", 
        "Service errors",
        "Service error_opportunity",
        "Receptions",
        "Reception_opportunity",
        "Sets",
        "Set_opportunity",
        "Set errors",
        "Set error_opportunity",
        "Attacks",
        "Attack_opportunity",
        "Attack errors",
        "Attack error_opportunity",
        "Aces",
        "Ace_opportunity",
        "Blocks",
        "Block_opportunity",
        "Digs",
        "Dig_opportunity",
        "First ball kills",
        "First ball kill_opportunity",
        "Kills",
        "Kill_opportunity"
        ])
    calc_stats(pd.read_csv("Data/test/test_game_1.csv"), players)
        
