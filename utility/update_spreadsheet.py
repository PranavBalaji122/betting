import requests
import json
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import os
from dotenv import load_dotenv
# load_dotenv()
from datetime import datetime, timedelta


def safe_int(value):
    """Parses value to int, returns None if parsing fails or if value is 'N/A'."""
    if value == "N/A":
        return None
    try:
        return int(value)
    except ValueError:
        return None

def extract_player_stats(summary_data):
    """
    Extracts player stats from the ESPN summary data. In addition to the default columns,
    this function adds 4 new fields to each player's 'Stats':
      - 'p_r'   = points + rebounds
      - 'p_a'   = points + assists
      - 'a_r'   = assists + rebounds
      - 'p_r_a' = points + rebounds + assists
    """
    if not summary_data:
        return None
    
    player_stats = []
    boxscore = summary_data.get("boxscore", {}).get("players", [])
    
    for team in boxscore:
        team_name = team.get("team", {}).get("displayName", "Unknown Team")
        
        for player in team.get("statistics", []):
            for athlete in player.get("athletes", []):
                player_name = athlete.get("athlete", {}).get("displayName", "Unknown Player")
                stats = athlete.get("stats", [])
                
                # These labels must match the order of values in 'stats'
                stat_labels = [
                    "Minutes", "Field Goals", "Three-Point Field Goals", "Free Throws",
                    "Offensive Rebounds", "Defensive Rebounds", "trb",
                    "ast", "Steals", "Blocks", "Turnovers", "Personal Fouls",
                    "Plus-Minus", "pts"
                ]
                
                # Create a dictionary of stats keyed by the above labels
                labeled_stats = {
                    stat_labels[i]: stats[i] if i < len(stats) else "N/A"
                    for i in range(len(stat_labels))
                }
                
                # Safely parse pts, trb, ast for combined fields
                pts_val = safe_int(labeled_stats.get("pts", "N/A"))
                trb_val = safe_int(labeled_stats.get("trb", "N/A"))
                ast_val = safe_int(labeled_stats.get("ast", "N/A"))

                def combine_stats(*vals):
                    """Helper to sum multiple safe_int values, returning 'N/A' if any is None."""
                    if any(v is None for v in vals):
                        return "N/A"
                    return str(sum(vals))

                # Add the new combined fields
                labeled_stats["p_r"]   = combine_stats(pts_val, trb_val)
                labeled_stats["p_a"]   = combine_stats(pts_val, ast_val)
                labeled_stats["a_r"]   = combine_stats(ast_val, trb_val)
                labeled_stats["p_r_a"] = combine_stats(pts_val, trb_val, ast_val)
                
                player_stats.append({
                    "Team": team_name,
                    "Player": player_name,
                    "Stats": labeled_stats
                })
    
    return player_stats


def get_nba_game_event_id(date):
    """
    Given a date in YYYYMMDD format, return a list of NBA event IDs from ESPN's scoreboard.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        events = data.get("events", [])
        if events:
            return [event["id"] for event in events]  # Returns a list of event IDs
    return []


def get_nba_game_summary(event_id):
    """
    Given an ESPN event_id, returns the detailed summary data (including boxscore) for that game.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def format_date(date_string):
    return f"{int(date_string[4:6])}/{int(date_string[6:])}/{date_string[2:4]}"




def updateGoogleSheet(column_range, date):
    """
    Updates Google Sheets using batch update to avoid hitting API limits.
    """
    SHEET_ID = os.getenv("SHEET_ID")  # The ID of your Google Sheet
    JSON_FILE = "json/nba_stats.json"  # The file with your players & stats
    CREDENTIALS_FILE = os.getenv("GOOGLE_API")  # Service account credentials

    # Define the Google API scopes
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]

    # Authorize with the service account
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)

    sheet_name = format_date(date)
    sheet = client.open_by_key(SHEET_ID).worksheet(sheet_name)

    # Load NBA stats from JSON file
    with open(JSON_FILE, "r") as file:
        nba_data = json.load(file)

    # Fetch headers and sheet data
    headers = sheet.get_values(f"A1:K1")[0]
    player_index = headers.index('Player')
    market_index = headers.index('Market')
    actual_column_letter = chr(64 + (headers.index('Actual') + 1))  # Convert index to column letter

    data = sheet.get_values(f"A{column_range[0]}:G{column_range[1]}")
    
    update_values = []  # List to collect values for batch update

    for i, row in enumerate(data):
        if len(row) < 2:
            update_values.append([""])  # If data is missing, append empty value
            continue

        player = row[player_index].strip()
        market = row[market_index].strip()

        # Find player stats
        player_dict = next((item for item in nba_data if item.get("Player") == player), None)
        if not player_dict:
            update_values.append([""])  # If no match, append empty value
            continue

        player_stats = player_dict.get("Stats", {})
        if market in player_stats:
            update_values.append([player_stats[market]])  # Append value as a list
        else:
            update_values.append([""])  # If no stat available, append empty value

    # Construct range in proper format (e.g., "H2:H75")
    update_range = f"{actual_column_letter}{column_range[0]}:{actual_column_letter}{column_range[1]}"

    # Batch update using a **single** API call
    if update_values:
        sheet.update(update_values, update_range, value_input_option="USER_ENTERED")
        print(f"Updated {len(update_values)} rows in range {update_range} in one batch request!")
    else:
        print("No matching stats found to update.")






def run(date, range_row):
    event_ids = get_nba_game_event_id(date)
    all_stats = []
    
    if event_ids:
        for event_id in event_ids:
            summary_data = get_nba_game_summary(event_id)
            stats = extract_player_stats(summary_data)
            if stats:
                all_stats.extend(stats)
            else:
                print(f"No stats available for game {event_id}")
        
        with open("JSON/nba_stats.json", "w") as file:
            json.dump(all_stats, file, indent=4)
        print("All stats saved to nba_stats.json")
    else:
        print("No games found for the given date.")

    updateGoogleSheet(range_row, date)


if __name__ == "__main__":
    range_row = (2,29) # range of rows to update
    date = (datetime.now() - timedelta(hours=20)).strftime("%Y%m%d")
    run(date, range_row)