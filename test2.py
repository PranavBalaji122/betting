
import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import numpy as np

def load_data1():
    df = pd.read_csv('github.csv')
    return df


def calculate_weights(days_since, decay_rate):
    """Calculate exponential weights where more recent games (higher days_since) have more influence."""
    max_days = np.max(days_since)
    return np.exp(decay_rate * (days_since - max_days))


# Function to predict points based on opponent and weighted recent games
def predict_features(df, player_id, opponent,feature):
    # Define columns of interest for similarity checking
    similarity_columns = [
        'mp','fg', 'fga', 'fg_percent', 'twop', 
        'twop_percent', 'threep', 'ft', 'ft_percent', 
        'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus','pts', 'hoa','p_r_a'
    ]
    similarity_columns.remove(feature)

    # Filter the data for the specific player
    player_data = df[df['player'] == player_id].copy()
    
    if player_data.empty:
        # print("No data available for this player.")
        return None

    # Further filter for games against the specified opponent
    opponent_data = player_data[player_data['opp'] == opponent].copy()
    
    if opponent_data.empty:
        # print("No historical games available against this opponent.")
        return None

    # Handle NaNs in the specified columns
    player_data_filtered = player_data[similarity_columns].fillna(player_data[similarity_columns].mean())
    player_data_filtered = player_data_filtered.fillna(0)

    # Scaling the specified columns between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(player_data_filtered)
    scaled_df = pd.DataFrame(scaled_data, columns=similarity_columns, index=player_data_filtered.index)
    decay_rate = 0.01 # Increase or decrease this to tune the time relevance
    weights = calculate_weights(player_data['days_since'], decay_rate)
    weighted_scaled_df = scaled_df.mul(weights, axis=0)  # Element-wise multiplication for weighting
    
    # Calculate the average of the weighted scaled statistics from games against the opponent
    specific_avg = weighted_scaled_df.loc[opponent_data.index].mean()

    # Calculate Euclidean distances from this average to all games within filtered player data
    distances = weighted_scaled_df.apply(lambda row: distance.euclidean(row, specific_avg), axis=1)
    player_data.loc[:,'distance'] = distances

    # Select the top 10 closest games based on the calculated distances
    closest_games = player_data.nsmallest(5, 'distance')

    print(closest_games['game_date'])

    # Calculate the predicted points by averaging the 'pts' of these closest games
    predicted_features = closest_games[feature].mean()
    return predicted_features

# Main function to run the prediction
def run(player, opp, feat):
    df = load_data1()
    player_id = player
    opponent = opp
    return predict_features(df, player_id, opponent, feat)

def main():
    print(run("Jayson Tatum","CHI", 'pts'))

if __name__ == '__main__':
    main()
