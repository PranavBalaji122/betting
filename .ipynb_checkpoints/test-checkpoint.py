import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

# Function to connect to the PostgreSQL database and load data
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

# Function to predict points based on opponent and the closest historical games
def predict_points(df, player_id, opponent):
    # Define columns of interest for similarity checking
    similarity_columns = [
        'result', 'total_score', 'mp', 'fga', 'fg_percent', 'twop', 
        'twop_percent', 'threep', 'ft', 'ft_percent', 'ts_percent', 
        'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc', 
        'bpm', 'plus_minus', 'p_r_a', 'p_r', 'p_a', 'a_r'
    ]

    # Filter the data for the specific player
    player_data = df[df['player'] == player_id]
    
    if player_data.empty:
        print("No data available for this player.")
        return None

    # Further filter for games against the specified opponent
    opponent_data = player_data[player_data['opp'] == opponent]
    
    if opponent_data.empty:
        print("No historical games available against this opponent.")
        return None

    # Select and handle NaNs in the specified columns
    player_data_filtered = player_data[similarity_columns]
    player_data_filtered = player_data_filtered.fillna(player_data_filtered.mean())  # Fill NaNs with column mean
    player_data_filtered = player_data_filtered.fillna(0)  # Fill remaining NaNs if any column was entirely NaN

    # Scaling the specified columns between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(player_data_filtered)
    scaled_df = pd.DataFrame(scaled_data, columns=similarity_columns, index=player_data_filtered.index)
    
    # Calculate the average of the scaled statistics from games against the opponent
    specific_avg = scaled_df.loc[opponent_data.index].mean()

    # Calculate Euclidean distances from this average to all games within filtered player data
    distances = scaled_df.apply(lambda row: distance.euclidean(row, specific_avg), axis=1)
    player_data_filtered['distance'] = distances

    # Reintegrate the distances back to the original player data and select the top 10 closest games
    player_data['distance'] = player_data_filtered['distance']
    closest_games = player_data.nsmallest(5, 'distance')

    # Calculate the predicted points by averaging the 'pts' of these closest games
    predicted_points = closest_games['pts'].mean()
    print(f"Predicted points based on the 10 closest games: {predicted_points}")
    return predicted_points

# Main function to run the prediction
def main():
    df = load_data()
    player_id = 'Jaylen Brown'  # Placeholder for player ID
    opponent = 'WAS'  # Placeholder for opponent code
    predict_points(df, player_id, opponent)

if __name__ == '__main__':
    main()
