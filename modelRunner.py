import json
from consienstyTest import getConsistency
from model import run
import pandas as pd
import psycopg2
import json

def load_odds(input_path):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return  
def load_injury_report():
    with open('injury.json', 'r') as file:
        rosters = json.load(file)
    player_set = set()
    for team_players in rosters.values():
        player_set.update(team_players)
    return player_set 

def get_consitnent_players(features):
    consistent_players = {}
    for feature in features:
        playerNames, teams = getConsistency(feature)
        consistent_players[feature] = playerNames
    return consistent_players


    
def calc_player_stats(odds_data, consistent_players):
    results = {}

    for entry in odds_data['DraftKings']:
        player = entry['player']
        team = entry['team']
        opponent = entry['opp']
        market = entry['market']
        line = entry['line']

        player_data = {
            'player': player,
            'market': market,
            'line': line,
            'bet': {}
        }

        # Check if the player is considered consistent in the current market
        if player in consistent_players.get(market, []):
            print(f"{player} is consistent in {market}")
            stat, error = run(player, team, opponent, 0, market,20)  # Assuming 'run' function returns a tuple (stat, error)

            # Decide and define the betting status
            is_good_bet = (stat < line and stat + error < line) or (stat > line and stat - error > line)
            bet_status = 'good' if is_good_bet else 'bad'

            player_data['bet'] = {
                'status': bet_status,
                'predicted': stat,
                'error': error
            }
            print(f"{player} is a {bet_status} bet")
        
            # Append only if there is a bet decision
            if player_data['bet']:  
                if team not in results:
                    results[team] = []
                results[team].append(player_data)

            

    return results     




def main():
    odds = load_odds('processed_odds.json')
    consitnent_players = get_consitnent_players(['pts', 'trb', 'ast','p_r_a'])
    injurys = load_injury_report()
    jsonData = calc_player_stats(odds, consitnent_players)
   

    filename = 'predctions.json'
    with open(filename, 'w') as file:
        json.dump(jsonData, file, indent=4)
    print(f"Data has been written to {filename}")   


if __name__ == '__main__':
    main()