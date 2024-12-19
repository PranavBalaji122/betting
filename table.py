import pandas as pd
import json

with open('filtered_odds.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Flatten the JSON data and prepare for DataFrame creation
rows = []
for bookie, bets in data.items():
    for bet in bets:
        player_vs_opp = f"{bet['player']} {bet['market']} vs {bet['opp']}"
        over_or_under = False if 'under' in bet else True
    
        
        rows.append({
            "Player vs Opponent": player_vs_opp,
            "Prediction": bet['prediction'],
            " ": "",  # Empty column
            "    ": "",  # Empty column
            "       ": "",  # Empty column
            "Line": bet['line'],
            "             ": "",  # Empty column
            "Our Guess": over_or_under,
        })

# Create DataFrame
df = pd.DataFrame(rows)

df.to_csv('guesses.csv', encoding='utf-8', index=False)
