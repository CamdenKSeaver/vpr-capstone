import pandas as pd

class Game:
    score : int
    teams : [str]
    player_stats : pd.DataFrame # Should contain 1 row per player
                                # player_name, id, team, stats for the game

                                #stats for game includes things like successful hits as well as opportunities to hit

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


    #Should be calculated in preprocessing
    def avg_team_attacks_per_opp_serve(self, team: str) -> int:
        #Should return the average number of attacks by a team after a serve
        pass

    def avg_team_attacks_per_opp_return(self, team: str) -> int:
        pass

    def avg_team_digs_per_opp_return(self, team: str) ->int:
        pass

    def avg_team_sets_per_opp_return(self, team: str) -> int:
        pass


    #should be calculated in preprocessing
    def avg_same_team_digs_per_dig(self, team : str) -> int:
        pass    

    def avg_same_team_digs_per_serve(self, team: str) -> int:
        #Should return the average number of attacks by a team after a serve
        pass

    def avg_same_team_sets_per_serve(self, team: str) -> int:
        #Should return the average number of attacks by a team after a serve
        pass

    def avg_same_team_sets_per_dig(self, team: str) -> int:
        pass

    def avg_same_team_hits_per_hit(self, team: str) -> int:
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
