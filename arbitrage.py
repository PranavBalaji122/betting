import requests
from datetime import datetime, timedelta, timezone
import json

#names of teams
nba_teams = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

def game_ids(base_url, api_key, commence_time_to):
    url = f"{base_url}?apiKey={api_key}&regions=us&oddsFormat=american&commenceTimeTo={commence_time_to}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return [game["id"] for game in response.json()]
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

def get_odds(base_url, game_id, api_key, market_types):
    market_data = {}
    for market_type in market_types:
        url = f"{base_url}/{game_id}/odds?apiKey={api_key}&markets={market_type}&oddsFormat=american&bookmakers=draftkings,fanduel,williamhill_us,betmgm"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                market_data[market_type] = data
            else:
                print(f"Failed to retrieve data for market {market_type}: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed for market {market_type}: {e}")
            continue
    return market_data

def collect_all_odds(base_url, api_key, game_ids):
    market_types = ["h2h,player_points", "player_rebounds", "player_assists", "player_threes","player_blocks", "player_steals", "player_points_rebounds_assists"]
    all_data = {}
    for game_id in game_ids:
        game_odds = get_odds(base_url, game_id, api_key, market_types)
        all_data[game_id] = get_odds
    return all_data

from datetime import datetime, timedelta


today = datetime.now()
tomorrow = today + timedelta(days=1)
tomorrow_at_5am = tomorrow.replace(hour=5, minute=0, second=0, microsecond=0)
iso_format_with_z = tomorrow_at_5am.isoformat() + 'Z'

commence_time_to = tomorrow_at_5am.isoformat() + 'Z'
api_key = 'a84fd2b2ffb360a0a568368830dc5295'
base_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
ids = game_ids(base_url, api_key, commence_time_to)
props = collect_all_odds(base_url, api_key, ids)

with open('props.json', 'w') as file:
    json.dump(props, file, indent=4)

print("Data saved to props.json")