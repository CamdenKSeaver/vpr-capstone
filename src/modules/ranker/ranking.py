import pandas as pd
from game import Game
from player import Player

def calc_score(game, original_player, new_player):
    #team,Season,Date,Team,Conference,Opponent Team,Opponent Conference,Location,Number,Player,P,S,MS,Kills,Errors,TotalAttacks,HitPct,Assists,Aces,SErr,Digs,RErr,BlockSolos,BlockAssists,BErr,TB,PTS,BHE,source_file,RetAtt,ContestID,is_away,kill_efficiency,Blocks

    #Find point difference from serves
        #Calculate serve change on digs / sets / hits / kills

    team = original_player.team
    opposing = game.against(original_player.id)
    point_change = 0


    #Serves  
    #This doen't really handle failing a serve twice
    #Find the Number of original player serves
    serve_chances = game.serve_runs(original_player.id)

    #Multiply by successful serve chance to get successful serves
    serves = serve_chances / (1 - new_player.serve_pct() * game.point_pct())

    #Mutliply by ace chance to get successful aces
    aces = serves * new_player.ace_pct()
    point_change += aces
    point_change -= game.aces(original_player.id)

    serve_diff = (serves - aces - game.serves(original_player.id) + game.aces(original_player.id))

    add_dig_chances_from_serves = game.avg_team_attacks_per_opp_serve(opposing) * serve_diff
    add_digs_from_serves = game.avg_team_digs_per_serve(team) * serve_diff   
    add_sets_from_serves = game.avg_team_sets_per_serve(team) * serve_diff


    #Returns
    #Find Return Attempts adds in additonal 
    return_chances = game.team_serves(opposing)
    #Multiply by return success percentage
    returns = return_chances * new_player.return_chance()

    returns_diff = (returns - game.returns(original_player.id))

    add_dig_chances_from_returns = game.avg_team_attacks_per_opp_return(opposing) * returns_diff
    add_digs_from_returns = game.avg_same_team_digs_per_opp_return(opposing) * returns_diff
    add_sets_from_returns = game.avg_same_team_sets_per_opp_return(opposing) * returns_diff


    #Digs
    #Find the number of attacks by the other team 
    dig_chances = game.team_attacks(opposing) 
    #Add additional attempts based on additional succesful serves and returns
    dig_chances += add_dig_chances_from_serves + add_dig_chances_from_returns
    #Multiply dig chances by dig success percent
    digs = dig_chances * new_player.dig_chance()

    digs_diff = (digs - game.digs(original_player.id))

    add_digs_from_digs = game.avg_same_team_digs_per_dig(team) * digs_diff
    add_sets_from_digs = game.avg_same_team_sets_per_dig(team) * digs_diff

    #Sets
    set_chances = game.team_digs(team)
    set_chances += add_digs_from_serves + add_digs_from_returns + add_digs_from_digs
    sets = set_chances * new_player.set_pct()
    assists = sets * new_player.assist_pct()
    point_change += assists - game.assists(original_player.id)

    sets_diff = sets - game.sets(original_player.id)

    #find point difference from hits/kills
    hit_chances = game.team_sets(team) + sets_diff + add_sets_from_digs + add_sets_from_returns + add_sets_from_serves
    hits = hit_chances * new_player.hit_pct()
    kills = hit_chances * new_player.kill_pct()

    avg_hit_chances_per_hit = game.avg_same_team_hits_per_hit()

    #Blocks need to be calculated

    kills += hits * avg_hit_chances_per_hit * new_player.kill_pct()
    point_change += kills - game.kills(original_player.id)

    return point_change

def rank_game(game, to_rank, other_players : list[Player]) -> int:
    #to_rank actually played the game
    #retreive current score
    actual_score : int = game.score()

    #for each other player 
    rel_score = 0
    count = 0
    for player in other_players:
        if player.id == to_rank.id:
            continue
        count += 1
        #compare to to_rank
        rel_score += calc_score(game, to_rank, player) - actual_score
    rel_score = rel_score / count

    #return score
    return rel_score

def rank_player(games, player, other_players) -> int:
    total = 0
    for game in games:
        total += rank_game(game, player, other_players)
    total = total / len(games)

    return total

def rank(games : list[Game], players: list[Player]):
    #get scores for players
    for player in players:
        players.score = rank_player(games, player, players)
    #sort players by score
    players.sort(key=lambda p: p.score)