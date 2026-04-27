import pandas as pd

class Player:
    name : str
    team : str
    id : str
    year : int
    data : pd.Dataframe 
    score : int

    '''
    These should all be data access operation with little calculation
    '''

    #Return : percentage of serves that are not errors
    def serve_pct(self) -> float:
        return 1 - self.serve_error_pct()
    
    #Return : percentage of serves that are erros
    def serve_error_pct(self) -> float:
        return self.data["id"==id]["serve_error_pct"]
    
    #Return : percentage of serves that are aces
    def ace_pct(self) -> float:
        return self.data["id"==id]["ace_pct"]
    
    #Return : given a successful serve by the enemy, chance that this player will return that serve
    def return_pct(self) -> float:
        return self.data["id"==id]["return_pct"]
    
    #Return : given a successful attack by the enemy, chance that this player will dig that attack
    def dig_pct(self) -> float:
        return self.data["id"==id]["dig_pct"]
    
    #Return : given a successful dig by a teamate, chance that this player will set that dig
    def set_pct(self) -> float:
        return self.data["id"==id]["set_pct"]
    
    #Return : given a successful dig by a teamate, chance that this player will assist the coming attack
    #Note : should be lower than return pct
    def assist_pct(self) -> float:
        return self.data["id"==id]["assist_pct"]
    
    #Return : given a successful set by a teammate, chance that this player will attack that set
    def attack_pct(self) -> float:
        return self.data["id"==id]["attack_pct"]
    
    #Return : given a successful set by a teammate, chance that this player will kill that set
    def kill_pct(self) -> float:
        return self.data["id"==id]["kill_pct"]