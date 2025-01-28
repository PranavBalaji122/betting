import pandas as pd

# Data provided by the user
data = {
        "DraftKings": [
            {
                "player": "Domantas Sabonis",
                "game": "SAC vs BKN",
                "market": "pts",
                "line": 19.5,
                "bet": {
                    "status": "good",
                    "predicted": 29.55,
                    "error": 6
                }
            },
            {
                "player": "Domantas Sabonis",
                "game": "SAC vs BKN",
                "market": "p_r_a",
                "line": 42.5,
                "bet": {
                    "status": "good",
                    "predicted": 51.55,
                    "error": 7
                }
            },
            {
                "player": "Devin Booker",
                "game": "PHO vs LAC",
                "market": "p_r_a",
                "line": 35.5,
                "bet": {
                    "status": "good",
                    "predicted": 48.65,
                    "error": 11
                }
            },
            {
                "player": "Jalen Duren",
                "game": "DET vs CLE",
                "market": "p_r",
                "line": 22.5,
                "bet": {
                    "status": "good",
                    "predicted": 29.3,
                    "error": 5
                }
            },
            {
                "player": "Jaylen Brown",
                "game": "BOS vs HOU",
                "market": "p_r",
                "line": 26.5,
                "bet": {
                    "status": "good",
                    "predicted": 36.1,
                    "error": 7
                }
            },
            {
                "player": "Alex Sarr",
                "game": "WAS vs DAL",
                "market": "p_r",
                "line": 20.5,
                "bet": {
                    "status": "good",
                    "predicted": 13.5,
                    "error": 3
                }
            },
            {
                "player": "Giannis Antetokounmpo",
                "game": "MIL vs UTA",
                "market": "p_r",
                "line": 43.5,
                "bet": {
                    "status": "good",
                    "predicted": 53.5,
                    "error": 6
                }
            },
            {
                "player": "Miles Bridges",
                "game": "CHO vs LAL",
                "market": "p_a",
                "line": 23.5,
                "bet": {
                    "status": "good",
                    "predicted": 32.45,
                    "error": 5
                }
            },
            {
                "player": "Devin Booker",
                "game": "PHO vs LAC",
                "market": "p_a",
                "line": 31.5,
                "bet": {
                    "status": "good",
                    "predicted": 45.2,
                    "error": 9
                }
            }
        ]
    }
df = pd.DataFrame([
    {
        "Player": entry["player"],
        "Market": entry["market"],
        "Predicted": entry["bet"]["predicted"],
        "Line": entry["line"],
        "Odds": entry["odds"]
    }
    for entry in data["DraftKings"]
])

# Save the DataFrame to a CSV file
df.to_csv('CSV/table.csv', encoding='utf-8', index=False)
