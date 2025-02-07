import pandas as pd
import json
# Data provided by the user
def write_table():
    data = {}
    with open('JSON/predictions.json', 'r') as file:
        data = json.load(file)

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


if __name__ == '__main__':
    write_table()