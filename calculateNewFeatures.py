from test import predict_feature
import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
def load_data():
    conn = psycopg2.connect(
        host="localhost",
        dbname="mnrj",
        user="postgres",
        password="gwdb",
        port=5600
    )
    df = pd.read_sql("SELECT * FROM nba;", conn)
    conn.close()
    return df

# Function to calculate new features 
# returns a datafram with all the new features
# we will use this dataframe as the X for the random regression forest model
def calulateNewFeatures(df, player_id, opponent):
    columns_to_predict = [
        'result', 'total_score', 'mp', 'fga', 'fg_percent', 'twop', 
        'twop_percent', 'threep', 'ft', 'ft_percent', 'ts_percent', 
        'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc', 
        'bpm', 'plus_minus', 'p_r_a', 'p_r', 'p_a', 'a_r'
    ]

    predictions = {}

    for feature in columns_to_predict:
        predicted_value = predict_feature(df, player_id, opponent, feature)
        predictions[feature] = [predicted_value]  # Store as a single-item list for DataFrame compatibility

    # Convert the dictionary to a DataFrame
    predicted_stats = pd.DataFrame(predictions)
    return predicted_stats
    


def main():
    df = load_data()
    player_id = 'Jaylen Brown'
    opponent = 'WAS'
    features = calulateNewFeatures(df, player_id, opponent)
    print(features)

if __name__ == '__main__':
    main()