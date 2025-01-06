import json


def load_data():
    try:
        with open('predctions.json', 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON data: {e}")
        return None

def evaluate_player_performance(data):
    correct_predictions = 0
    good_bet_count = 0
    if data:
        for team, players in data.items():
            for player in players:
                bet_status = player['bet']['status']
                if bet_status == "good":
                    player_name = player['player']
                    market = player['market']
                    line = player['line']
                    prediction = player['bet']['predicted']
                    actual_score = int(input(f"How many {market} did {player_name} score today? "))
                    if (prediction < line and actual_score < line) or (prediction > line and actual_score > line):
                        correct_predictions += 1
                    good_bet_count += 1
        if good_bet_count > 0:
            return 100 * (correct_predictions / good_bet_count), correct_predictions
        else:
            return 0, 0
    return 0, 0




if __name__ == '__main__':
    data = load_data()
    percentage, correct_predictions = evaluate_player_performance(data)
    print(f"Accuracy: {percentage}%, Correct Predictions: {correct_predictions}")
