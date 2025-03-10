import pandas as pd
import json
# Data provided by the user
def write_table(filename='json/predictions.json'):
    data = {}
    with open(filename, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame([
        {
            "Player": entry["player"],
            "Market": entry["market"],
            "Predicted": round(float(entry["bet"]["predicted"])),
            "Buffer": round(float(entry["bet"]["error"])),
            "Line": entry["line"],
            #"Odds": entry["odds"],
            "Last Ten": entry["last_ten"],
            "Game": entry["game"]
        }
        for entry in data["DraftKings"]
    ])

    # Save the DataFrame to a CSV file
    df.to_csv('csv/table.csv', encoding='utf-8', index=False)
# Mapping for market abbreviations
    market_mapping = {
        "pts": "PTS",
        "trb": "REB",
        "ast": "AST",
        "p_r": "P+R",
        "p_a": "P+A",
        "p_r_a": "P+R+A",
        "a_r": "A+R"
    }


    output_lines = []
    for entry in data.get("DraftKings", []):
        player = entry.get("player", "Unknown")
        line_value = entry.get("line", 0)
        odds = entry.get("odds", "")
        game = entry.get("game", "Unknown Game")
        bet_info = entry.get("bet", {})
        predicted = bet_info.get("predicted", 0)
        market_key = entry.get("market", "").lower()

        # Determine if the prediction is under or over the line.
        direction = "Under" if predicted < line_value else "Over"
        
        # Map market value; default to upper-case if not found in mapping.
        market_formatted = market_mapping.get(market_key, market_key.upper())
        
        # Format the output line as required
        formatted_line = f"🏀 {game}: {player} {direction} {line_value} {market_formatted}  ({odds} DK)"
        output_lines.append(formatted_line)

    # Write the formatted lines to a text file (e.g., 'output.txt')
    with open('output.txt', 'w') as f:
        for line in output_lines:
            f.write(line + "\n")


if __name__ == '__main__':
    write_table()