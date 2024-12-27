import json
from consienstyTest import getConsistency
from model import run
import pandas as pd
import psycopg2
import json


def load_data():
    try:
        with open('predctions.json', 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return

def evaluate_player_performance(data):
    correct_predictions = 0
    count = 0
    for team, players in data.items():
        for player in players:
            player_name = player['player']
            market = player['market']
            line = player['line']
            prediction = player['bet']['predicted']
            actual_score = int(input(f"How many {market} did {player_name} score today? "))
            if prediction < line and actual_score < line:
                correct_predictions += 1
            elif prediction > line and actual_score > line:
                correct_predictions += 1
            count += 1
    
    return 100 *(correct_predictions / count), correct_predictions











if __name__ == '__main__':
    data = load_data()
    print(evaluate_player_performance(data))
    