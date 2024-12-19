import json
import numpy as np
from test import run 

def process_json_data(input_path, output_path):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return  # Stop processing if JSON is not valid

    results = {key: [] for key in data.keys()}
    
    for platform, props_list in data.items():
        for prop in props_list:
            player = prop['player']
            opp = prop['opp']
            line = prop['line']
            market = prop['market']
            prediction = run(player, opp, market)
            
            if prediction is None:
                continue

            if (line - prediction > 2) and 'under' in prop:
                modified_prop = {
                    'player': player,
                    'team': prop['team'],
                    'opp': opp,
                    'hoa': prop['hoa'],
                    'line': line,
                    'market': market,
                    'under': prop['under']
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
