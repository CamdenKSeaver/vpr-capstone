import pandas as pd

class Game:
    score : int
    teams : list[str]
    player_data : pd.DataFrame # Should contain 1 row per player
                                # player_name, id, team, stats for the game

                                #stats for game includes things like successful hits as well as opportunities to hit
    #Number of sets made after a serve by the same team
    same_team_sets_from_serve : dict = {}
    opp_team_attacks_from_serve : dict = {}
    opp_team_attacks_from_return : dict = {}

    def __init__(self, game_data : pd.DataFrame, away_team_players, home_team_players):
        away_team = game_data.iloc[0]["away_team"]
        home_team = game_data.iloc[0]["home_team"]
        teams = [away_team, home_team]

        #I am sure there is still the possiblity of a key errror here
        self.player_data = pd.DataFrame(columns=[
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

        for player in away_team_players:
            row = {col: 0.0 for col in self.player_data.columns}
            row["player"] = player
            row["Team"] = away_team

            self.player_data.loc[len(self.player_data)] = row

        for player in home_team_players:
            row = {col: 0.0 for col in self.player_data.columns}
            row["player"] = player
            row["Team"] = home_team

            self.player_data.loc[len(self.player_data)] = row

        self.same_team_sets_from_serve[away_team] = 0
        self.same_team_sets_from_serve[home_team] = 0

        self.opp_team_attacks_from_serve[away_team] = 0
        self.opp_team_attacks_from_serve[home_team] = 0

        self.opp_team_attacks_from_return[away_team] = 0
        self.opp_team_attacks_from_return[home_team] = 0

    

    #Can be calculated at runtime
    def against(self, id) -> str:
        #Should return the team the player is not playing for.
        pass

    def point_pct(self, team) -> float:
        pass

    def score(self) -> str:
        return self.score


    #Can be retreived at runtime
    def team_serves(self, team: str) -> int:
        #Return the number of serves by a team
        pass

    def team_attacks(self, team: str) -> int:
        #Should return the total number of attacks by a team
        pass

    def team_digs(self, team: str) -> int:
        pass

    def team_sets(self, team: str) -> int:
        pass

    def team_returns(self, team: str)-> int:
        pass


    #Should be calculated in preprocessing
    def avg_team_attacks_per_opp_serve(self, team: str) -> float:
        #Should return the average number of attacks by a team after a serve
        self.opp_team_attacks_from_serve[team] / self.team_serves(self.against(team))

    def avg_team_attacks_per_opp_return(self, team: str) -> float:
        self.opp_team_attacks_from_return[team] / self.team_returns(self.against(team))

    def avg_team_digs_per_opp_return(self, team: str) ->float:
        return 5

    def avg_team_sets_per_opp_return(self, team: str) -> float:
        return 4


    #should be calculated in preprocessing
    def avg_same_team_digs_per_dig(self, team : str) -> float:
        return 1.4    

    def avg_same_team_digs_per_serve(self, team: str) -> float:
        #Should return the average number of attacks by a team after a serve
        pass

    def avg_same_team_sets_per_serve(self, team: str) -> float:
        #Should return the average number of attacks by a team after a serve
        return self.same_team_sets_from_serve[team] / self.team_serves(team)

    def avg_same_team_sets_per_dig(self, team: str) -> float:
        pass

    def avg_same_team_hits_per_hit(self, team: str) -> float:
        pass
    

    #Should be calculated in preprocessing
    def serves(self, id) -> int:
        #return the number of serve attempts, not including second chances for an error of the relevant player
        return 1
    
    def serve_runs(self, id) -> int:
        pass 

    def aces(self, id) -> int:
        pass

    def returns(self, id) -> int:
        pass

    def digs(self, id) -> int:
        pass

    def sets(self, id) -> int:
        pass

    def assists(self, id) -> int:
        pass

    def hits(self, id) -> int:
        return self.player_stats["id"==id]["hits"]
    
    def kills(self, id) -> int:
        pass
