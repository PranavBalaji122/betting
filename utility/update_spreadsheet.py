import requests
import json
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials


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



def updateGoogleSheet(column_range):
    """
    column_range: A tuple (start_row, end_row) for the Google Sheet rows
                  you want to process, inclusive or exclusive depending on usage.
                  For example, (1, 10) means "read from row 1 to row 10".
    """
    SHEET_ID = "1lbNo8exL_KWPb05pVXKD5IhZuWbirjSup3XQTtW4HBU"
    JSON_FILE = "JSON/nba_stats.json"       # The file with your players & stats
    CREDENTIALS_FILE = "JSON/nba_creds.json"  # Service account credentials

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

    # Open the sheet (worksheet named "table")
    sheet = client.open_by_key(SHEET_ID).worksheet("Sheet4")

    # Load NBA stats from JSON file
    with open(JSON_FILE, "r") as file:
        nba_data = json.load(file)

    # Fetch the rows from the sheet
    # For example, if column_range = (2, 10), it fetches rows 2 to 10 (columns A to G).
    data = sheet.get_values(f"A{column_range[0]}:G{column_range[1]}")

    print("Retrieved Data from Google Sheets:")
    # Let's list the rows we just fetched
    for i, row in enumerate(data):
        # The actual row in Google Sheets is offset from `column_range[0]`
        sheet_row_number = column_range[0] + i
        if row:
            print(f"Sheet row {sheet_row_number}: {row}")
        else:
            print(f"Sheet row {sheet_row_number}: EMPTY")

    # Now, go through each row and update the sheet if there's a matching stat
    for i, row in enumerate(data):
        sheet_row_number = column_range[0] + i

        # We expect at least 2 columns: [Player, Market]
        if len(row) < 2:
            continue

        # Extract player name and stat market (e.g., "Points", "Assists", etc.)
        player = row[1].strip()
        market = row[2].strip()

        # Debug: show what we're trying to match
        print(f"\nChecking row {sheet_row_number} => Player: '{player}', Market: '{market}'")

        # Attempt to find the player's Stats dict in the JSON
        # Each element in nba_data is expected to be:
        # {
        #   "Team": "<team name>",
        #   "Player": "<player name>",
        #   "Stats": { "Points": "10", "Assists": "2", ... }
        # }
        player_dict = next((item for item in nba_data if item.get("Player") == player), None)
        
        if player_dict is None:
            print("  -> No player match found in JSON.")
            continue

        player_stats = player_dict.get("Stats", {})
        if not player_stats:
            print("  -> 'Stats' key not found or empty for this player in JSON.")
            continue

        # Now we see if the requested market is in that player's stats
        if market in player_stats:
            stat_value = player_stats[market]
            print(f"  -> Updating row {sheet_row_number}, column G with value: {stat_value}")
            sheet.update_cell(sheet_row_number, 9, stat_value)
        else:
            print(f"  -> Market '{market}' not found in player_stats keys: {list(player_stats.keys())}")

    print("\nGoogle Sheet updated successfully!")


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

    updateGoogleSheet(range_row)


if __name__ == "__main__":
    range_row = (2,40) # range of rows to update
    date = "20250211" # Change this to the desired date (YYYYMMDD)
    run(date, range_row)