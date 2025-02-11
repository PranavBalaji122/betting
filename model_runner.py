import json
from collections import defaultdict
from utility.consistentsyTest import getConsistency
from models.model_rf import run_rf
from models.model_gb import run_gb
from utility.pred_table import write_table
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text

def load_odds(input_path):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return  

def load_injury_report():
    with open('JSON/injury.json', 'r') as file:
        rosters = json.load(file)
    
    player_set = set()
    for team_players in rosters.values():
        # team_players is a list of dicts like {"player": "...", "status": "..."}
        for player_info in team_players:
            player_set.add(player_info["player"])
    
    return player_set

def get_consistent_players(features):
    consistent_players = {}
    for feature in features:
        playerNames, teams = getConsistency(feature)
        consistent_players[feature] = playerNames
    return consistent_players

def get_player_last(player, market, line):
    # Load the CSV into a DataFrame
    df = pd.read_csv('CSV/sql.csv')
    
    # Filter rows for the specified player
    # Adjust 'player_name' if your CSV column is named differently
    player_df = df[df['player'] == player].copy()
    
    if player_df.empty:
        return 0
    player_df.sort_values(by='date', ascending=False, inplace=True)
    last_ten = player_df.head(10)
    
    over_count = (last_ten[market] > line).sum()
    
    return f"{int(over_count)}/10"

def calc_player_stats(odds_data, consistent_players,injuries):
    conn = create_engine('postgresql+psycopg2://postgres:gwdb@localhost:5600/mnrj')
    results = defaultdict(list)
    for bookmaker_name, entries in odds_data.items():  # This will give you each bookmaker's name and their entries
        for entry in entries:
            try:
                player = entry['player']
                team = entry['team']
                opponent = entry['opp']
                market = entry['market']
                hoa = entry['hoa']
                line = float(entry['line'])  # Ensure line is treated as a float for comparison
                
                if(market in ['pts','p_r_a', 'p_r', 'p_a']):
                    line = line - 3
                elif (market != 'a_r'):
                    line = line - 1


                player_data = {
                    'player': player,
                    'game': f"{team} vs {opponent}",
                    'market': market,
                    'line': line,
                    'over': entry["over"],
                    'under': entry["under"],
                    'bet': {}
                }
                if player in consistent_players.get(market, []):
                    if player not in injuries:
                        print(f"Running model on {player} for {market}")
                        # stat, error = run_rf(player, team, opponent, hoa, market,20) 
                        stat, error = run_gb(player, team, opponent, hoa, market,20)
                        buffer = error  # or however buffer is determined

                        is_good_bet = ((stat < line and (line > (stat + (buffer*0.8)))) or (stat > line and ((stat - (buffer*0.8)) > line))) or (market in ['pts','p_r_a', 'p_r', 'p_a'] and buffer < 4)
                        is_good_bet = True
                        bet_status = 'good' if is_good_bet else 'bad'

                        player_data['bet'] = {
                            'status': bet_status,
                            'predicted': stat,
                            'error': error
                        }
                        if(stat < line):
                            del player_data['over']
                            player_data['odds'] = player_data.pop('under')
                        else:
                            del player_data['under']
                            player_data['odds'] = player_data.pop('over')
                        player_data['last_ten']= get_player_last(player,market,line)
                        # Add to results only if it's a good bet and meets the buffer criteria
                        if bet_status == "good":
                            results[bookmaker_name].append(player_data)


            except Exception as e:
                print(f"Error processing data for player {player}: {e}")

    return results

def main():

    odds = load_odds('JSON/processed_odds.json')
    consitnent_players = get_consistent_players(['pts', 'trb', 'ast','p_r','p_a','a_r','p_r_a'])
    injuries = load_injury_report()
    jsonData = calc_player_stats(odds, consitnent_players, injuries)


    filename = 'JSON/predictions.json'
    with open(filename, 'w') as file:
        json.dump(jsonData, file, indent=4)
    print(f"Data has been written to {filename}") 
    write_table() 

if __name__ == '__main__':
    main()