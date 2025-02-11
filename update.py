import requests
from datetime import datetime, timedelta, timezone
from utility.process_props import process_props
from utility.load_injuries import load_injuries
from utility.get_new_data import get_new_Data
from utility.pipeline import run_pipeline
import json
import os

api_key = os.getenv("ODDS_KEY")

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
        url = f"{base_url}/{game_id}/odds?apiKey={api_key}&markets={market_type}&oddsFormat=american&bookmakers=draftkings"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for bookmaker in data["bookmakers"]:
                    if bookmaker["title"] not in market_data:
                        market_data[bookmaker["title"]] = {}
                    if market_type not in market_data[bookmaker["title"]]:
                        market_data[bookmaker["title"]][market_type] = []
                    for market in bookmaker["markets"]:
                        for outcome in market["outcomes"]:
                            market_data[bookmaker["title"]][market_type].append({
                                "description": outcome["description"],
                                "home_team": nba_teams.get(data["home_team"]),
                                "away_team": nba_teams.get(data["away_team"]),
                                "name": outcome["name"],
                                "point": outcome.get("point", None),
                                "price": outcome["price"],
                                "game_id": game_id
                            })
            else:
                print(f"Failed to retrieve data for market {market_type}: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed for market {market_type}: {e}")
            continue
    return market_data

def collect_all_odds(base_url, api_key, game_ids):
    market_types = ["player_points", "player_rebounds", "player_assists", "player_points_rebounds_assists", "player_points_rebounds", "player_points_assists", "player_rebounds_assists"]
    # market_types = ["player_points", "player_rebounds","player_assists", "player_points_rebounds_assists"]
    # market_types = ["player_points", "player_rebounds", "player_assists"]
    all_bookmakers_data = {}

    for game_id in game_ids:
        game_odds = get_odds(base_url, game_id, api_key, market_types)
        for bookmaker, markets in game_odds.items():
            if bookmaker not in all_bookmakers_data:
                all_bookmakers_data[bookmaker] = {market_type: [] for market_type in market_types}
            for market_type, data in markets.items():
                all_bookmakers_data[bookmaker][market_type].extend(data)
    return all_bookmakers_data

def updates():

    get_new_Data()
    run_pipeline()

    from datetime import datetime, timedelta


    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    tomorrow_at_5am = tomorrow.replace(hour=5, minute=0, second=0, microsecond=0)

    commence_time_to = tomorrow_at_5am.isoformat() + 'Z'
    base_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    ids = game_ids(base_url, api_key, commence_time_to)
    props = collect_all_odds(base_url, api_key, ids)
    
    with open('json/props.json', 'w') as file:
        json.dump(props, file, indent=4)

    process_props()
    load_injuries()

if __name__ == "__main__":
    updates()