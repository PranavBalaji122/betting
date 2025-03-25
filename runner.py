import json
from collections import defaultdict
from utility.consistentsyTest import getConsistency
from models.soft_predictor import soft
from models.model_rf import run_rf
from models.model_xgb import run_xgb
from utility.pred_table import write_table
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv
load_dotenv()

banned_list = [
    "Aaron Nesmith",
    "Joel Embiid",
    "Tyrese Maxey",  
    "Ty Jerome",
    "Lauri Markkanen", 
    "Khris Middleton",
    "Royce O'Neale",
    "Bilal Coulibaly",
    "Jalen Duren",
    "Markelle Fultz",
]

def load_odds(input_path):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return  

def load_injury_report():
    with open('json/injury.json', 'r') as file:
        rosters = json.load(file)
    
    player_set = set()
    for team_players in rosters.values():
        for player_info in team_players:
            player_set.add(player_info["player"])
    
    return player_set

def get_consistent_players(features):
    consistent_players = {}
    for feature in features:
        playerNames, teams = getConsistency(feature)
        consistent_players[feature] = playerNames
    return consistent_players

def get_rank(player, consistent_players, market):
    return (consistent_players.get(market, [])).index(player)

def get_player_last(player, market, line, stat=-1):
    df = pd.read_csv('csv/sql.csv')
    player_df = df[df['player'] == player].copy()

    # Slight tweak to 'line' if it's tpm or something else
    if market == "tpm":
        line = line - 0.4
    elif market in ('pts','trb','ast','a_r'): 
        line = line - 1
    else:
        line = line - 2

    if player_df.empty:
        return 0

    player_df.sort_values(by='date', ascending=False, inplace=True)
    last_ten = player_df.head(10)

    # Decide if we want (stat >= line) or (stat < line)
    if (stat == -1 or stat >= line):
        count = (last_ten[market] >= line).sum()
    else:
        count = (last_ten[market] < line).sum()
    return count

def calc_player_stats(odds_data, consistent_players, injuries):
    conn = create_engine(os.getenv("SQL_ENGINE"))
    results = defaultdict(list)

    for bookmaker_name, entries in odds_data.items():
        for entry in entries:
            try:
                player = entry['player']
                team = entry['team']
                opponent = entry['opp']
                market = entry['market']
                hoa = entry['hoa']
                line = float(entry['line'])
                
                player_data = {
                    'player': player,
                    'game': f"{team} vs {opponent}",
                    'market': market,
                    'line': line,
                    'over': entry["over"],
                    'under': entry["under"],
                    'bet': {}
                }

                # --- Old logic: get last-10 (optional to keep) ---
                last_ten = get_player_last(player, market, line)
                
                # If you want to remove or loosen these conditions, go ahead.
                if (
                    last_ten > 6
                    and player in consistent_players.get(market, [])
                    and player not in injuries
                    and player not in banned_list
                ):
                    print(f"Running models on {player} for {market}")
                    
                    # Run both models
                    #stat_rf, err_rf = run_rf(player, team, opponent, hoa, market, 20)
                    stat_gb, err_gb = run_xgb(player, team, opponent, hoa, market, 20)
                    
                    # Example tweak for tpm error
                    if market == "tpm":
                        err_rf *= 0.8
                        err_gb *= 0.8

                    #rf_says_over = (stat_rf > line + err_rf)
                    gb_says_over = (stat_gb > line + err_gb*0.9)

                    final_pred = None
                    final_err = None
                    # Just combining logic as you had
                    # if rf_says_over and gb_says_over:
                    #     final_pred = (stat_rf + stat_gb) / 2.0
                    #     final_err = (err_rf + err_gb) / 2.0
                    # elif rf_says_over:  # only RF
                    #     final_pred = stat_rf
                    #     final_err  = err_rf
                    # elif gb_says_over:  # only GB
                    #     final_pred = stat_gb
                    #     final_err  = err_gb
                    
                    if gb_says_over:
                        final_pred = stat_gb
                        final_err  = err_gb

                    if final_pred is None:
                        continue  # skip if no "over" scenario from the models

                    # ----------------------------
                    # New metric: scale or “score”
                    # (predicted - error - line)/line
                    # ----------------------------
                    score = (final_pred - final_err - line) / line
                    player_data['bet'] = {
                        'predicted': round(float(final_pred), 3),
                        'error': round(float(final_err), 2),
                        'score': round(float(score), 3),
                        'rank': get_rank(player, consistent_players, market)
                    }

                    # If you want to skip if score < 0, do so:
                    if score < 0:
                        # Means we’re not comfortable it goes over
                        continue

                    # old logic to pick "over" or "under" odds 
                    # (fyi, with the new 'score' approach, you might just do over)
                    if stat_gb >= line:
                        # "over" scenario
                        del player_data['under']
                        player_data['odds'] = player_data.pop('over')
                    else:
                        del player_data['over']
                        player_data['odds'] = player_data.pop('under')

                    # Store the last-10
                    player_data['last_ten_val'] = int(last_ten)
                    player_data['last_ten'] = f"{int(last_ten)}/10"
                    results[bookmaker_name].append(player_data)

            except Exception as e:
                print(f"Error processing data for player {player}: {e}")

    return results

def main():
    odds = load_odds('json/processed_odds.json')
    consistent_players = get_consistent_players(['pts','trb','ast','p_r','p_a','a_r','p_r_a','tpm'])
    injuries = load_injury_report()
    jsonData = calc_player_stats(odds, consistent_players, injuries)

    # # Condense or group bets for the same player, etc.
    # for bookmaker_name, bets in jsonData.items():
    #     grouped_by_player = defaultdict(list)
    #     for bet_info in bets:
    #         grouped_by_player[bet_info['player']].append(bet_info)
    
    #     final_bets_for_bookmaker = []
    #     for player, player_bets in grouped_by_player.items():
    #         # Example: keep the bet with the highest score, rather than last_ten
    #         top_bet = max(player_bets, key=lambda x: x['bet']['score'])
    #         final_bets_for_bookmaker.append(top_bet)

    #     jsonData[bookmaker_name] = final_bets_for_bookmaker

    filename = 'json/predictions.json'
    with open(filename, 'w') as file:
        json.dump(jsonData, file, indent=4)
    print(f"Data has been written to {filename}")
    write_table('json/predictions.json')

if __name__ == '__main__':
    main()
