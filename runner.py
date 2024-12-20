import json
import numpy as np
from test import run 
DIFF = 3

def load_injury_report():
    with open('injury.json', 'r') as file:
        rosters = json.load(file)
    player_set = set()
    for team_players in rosters.values():
        player_set.update(team_players)
    return player_set

def process_json_data(input_path, output_path):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return  # Stop processing if JSON is not valid

    results = {key: [] for key in data.keys()}
    injury = load_injury_report()
    for platform, props_list in data.items():
        for prop in props_list:
            player = prop['player']
            if(player in injury):
                continue
            opp = prop['opp']
            line = prop['line']
            market = prop['market']
            prediction = run(player, opp, market)
            
            if prediction is None:
                continue

            if (line - prediction > DIFF):
                modified_prop = {
                    'player': player,
                    'team': prop['team'],
                    'opp': opp,
                    'hoa': prop['hoa'],
                    'line': line,
                    'market': market,
                    'under': prop['under'],
                    'prediction': prediction
                }
                results[platform].append(modified_prop)
            elif (prediction - line > DIFF):
                modified_prop = {
                    'player': player,
                    'team': prop['team'],
                    'opp': opp,
                    'line': line,
                    'market': market,
                    'over': prop['over'],
                    'prediction': prediction
                }
                results[platform].append(modified_prop)
                
            
            

    try:
        with open(output_path, 'w') as file:
            json.dump(results, file, indent=4)
    except IOError as e:
        print(f"Error writing to file: {e}")

    print(f"Filtered data written to {output_path}")

# Example usage
process_json_data('processed_odds.json', 'filtered_odds.json')
