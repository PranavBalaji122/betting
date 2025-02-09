import pandas as pd
import json

def write_table():
    # Load the JSON data
    with open('JSON/predictions.json', 'r') as file:
        data = json.load(file)

    # Define the list of words to exclude
    exclude_words = ["HOU", "DAL", "LAL", "IND"]

    # Filter out entries where the 'game' field contains any of the excluded words
    filtered_entries = [
        entry for entry in data["DraftKings"]
        if not any(word in entry.get("game", "") for word in exclude_words)
    ]

    # Create a DataFrame from the filtered entries
    df = pd.DataFrame([
        {
            "Player": entry["player"],
            "Market": entry["market"],
            "Predicted": entry["bet"]["predicted"],
            "Line": entry["line"],
            "Odds": entry["odds"],
            "Game": entry["game"]
            
        }
        for entry in filtered_entries
    ])

    # Save the DataFrame to a CSV file
    df.to_csv('CSV/table1.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    write_table()
