import pandas as pd

# Data provided by the user
data = {
    "DraftKings": [
        {
            "player": "Giannis Antetokounmpo",
            "game": "MIL vs POR",
            "market": "p_r",
            "line": 44.5,
            "bet": {
                "status": "good",
                "predicted": 53.35,
                "error": 5
            }
        },
        {
            "player": "Damian Lillard",
            "game": "MIL vs POR",
            "market": "p_a",
            "line": 34.5,
            "bet": {
                "status": "good",
                "predicted": 44.55,
                "error": 8
            }
        },
        {
            "player": "Anthony Davis",
            "game": "LAL vs PHI",
            "market": "a_r",
            "line": 15.5,
            "bet": {
                "status": "good",
                "predicted": 19.95,
                "error": 3
            }
        },
        {
            "player": "Anfernee Simons",
            "game": "POR vs MIL",
            "market": "p_r",
            "line": 22.5,
            "bet": {
                "status": "good",
                "predicted": 32.85,
                "error": 8
            }
        },
        {
            "player": "Toumani Camara",
            "game": "POR vs MIL",
            "market": "p_r",
            "line": 17.5,
            "bet": {
                "status": "good",
                "predicted": 24.6,
                "error": 5
            }
        },
        {
            "player": "Deni Avdija",
            "game": "POR vs MIL",
            "market": "p_r_a",
            "line": 24.5,
            "bet": {
                "status": "good",
                "predicted": 33.7,
                "error": 8
            }
        },
    ]
}
df = pd.DataFrame([
    {
        "Player": entry["player"],
        "Market": entry["market"],
        "Predicted": entry["bet"]["predicted"],
        "Line": entry["line"]
    }
    for entry in data["DraftKings"]
])

# Save the DataFrame to a CSV file
df.to_csv('CSV/table.csv', encoding='utf-8', index=False)
