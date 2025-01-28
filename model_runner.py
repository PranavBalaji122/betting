import json
from collections import defaultdict
from utility.consistentsyTest import getConsistency
from models.model import run
import pandas as pd
import psycopg2
import json

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
        player_set.update(team_players)
    return player_set 

def get_consistent_players(features):
    consistent_players = {}
    for feature in features:
        playerNames, teams = getConsistency(feature)
        consistent_players[feature] = playerNames
    return consistent_players


    
def calc_player_stats(odds_data, consistent_players):
    results = defaultdict(list)
    for bookmaker_name, entries in odds_data.items():  # This will give you each bookmaker's name and their entries
        for entry in entries:
            try:
                player = entry['player']
                team = entry['team']
                opponent = entry['opp']
                market = entry['market']
                line = float(entry['line'])  # Ensure line is treated as a float for comparison

                player_data = {
                    'player': player,
                    'game': f"{team} vs {opponent}",
                    'market': market,
                    'line': line,
                    'bet': {}
                }

                if player in consistent_players.get(market, []):
                    print(f"Running model on {player} for {market}")
                    stat, error = run(player, team, opponent, 0, market, 20)  # Ensure the run function is defined
                    buffer = error  # or however buffer is determined
                    
                    is_good_bet = (stat < line and (line - (stat + buffer)) > 2.5) or (stat > line and ((stat - buffer) - line) > 2.5)
                    bet_status = 'good' if is_good_bet else 'bad'

                    player_data['bet'] = {
                        'status': bet_status,
                        'predicted': stat,
                        'error': error
                    }

                    # Add to results only if it's a good bet and meets the buffer criteria
                    if bet_status == "good":
                        results[bookmaker_name].append(player_data)
            except Exception as e:
                print(f"Error processing data for player {player}: {e}")

    return results


def main():
    odds = load_odds('JSON/processed_odds.json')
    consitnent_players = get_consistent_players(['pts', 'trb', 'ast','p_r','p_a','a_r','p_r_a'])
    injurys = load_injury_report()
    jsonData = calc_player_stats(odds, consitnent_players)
   

    filename = 'JSON/predctions.json'
    with open(filename, 'w') as file:
        json.dump(jsonData, file, indent=4)
    print(f"Data has been written to {filename}")   


if __name__ == '__main__':
    main()