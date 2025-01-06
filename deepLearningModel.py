import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import StringIO
import json
import psycopg2
from sqlalchemy import create_engine
from psycopg2.extras import Json
from psycopg2 import sql
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import math, statistics
from test import soft

class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, 32)          # Adjust accordingly
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)                   # Applying dropout to the first layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                       # No activation function in the output layer for regression
        return x


def load_nba(player, conn):
    try:
        query = f"SELECT * FROM nba WHERE player = '{player}';"
        df = pd.read_sql(query, conn)
        return df
    except psycopg2.Error as e:
        print(f"Error occurred while connecting to the database or executing query: {e}")
        return None   
def load_player_positions(conn):
    try:
        # Use the connection to execute the query
        query = f"SELECT * FROM latest_player_teams;"
        df = pd.read_sql(query, conn)
        return df
    except psycopg2.Error as e:
        print(f"Error occurred while connecting to the database or executing query: {e}")
        return None
def load_game_stats(player,conn):
    positions_df = load_player_positions(conn)
    if positions_df is None:
        return pd.DataFrame()

    try:
        query = f"""
        SELECT *
        FROM game_stats
        WHERE teammates_points::jsonb ? '{player}';  -- Checking if player key exists in JSON
        """
        df = pd.read_sql(query, conn)
    
        def aggregate_position_data(json_data, exclude_player):
            if exclude_player in json_data:
                del json_data[exclude_player]  # Remove the player from the data
            
            position_totals = {'G': 0, 'F': 0, 'C': 0}
            for player, value in json_data.items():
                pos = positions_df.loc[positions_df['player'] == player, 'pos'].values
                if pos.size > 0:
                    position_totals[pos[0]] += value

            return position_totals
        
        stats_fields = ['teammates_points', 'teammates_rebounds', 'teammates_assists', 'opponents_points', 'opponents_rebounds', 'opponents_assists',
                        'teammates_pr','teammates_pa','teammates_ar','opponents_pr','opponents_pa','opponents_ar','teammates_pra','opponents_pra',
                        'teammates_blocks', 'teammates_turnovers', 'opponents_blocks', 'opponents_turnovers']
        # Applying the transformation for each stats field
        for stat_field in stats_fields:
            df[stat_field] = df[stat_field].apply(lambda x: aggregate_position_data(x, player))
        
        for field in stats_fields:
            # Expand each dictionary into separate columns
            df_field = df[field].apply(pd.Series)
            df_field.columns = [f"{field}_{col}" for col in df_field.columns]  # Rename columns to include stat field
            df = pd.concat([df, df_field], axis=1)
            df.drop(field, axis=1, inplace=True)  # Drop the original column


        return df
    except psycopg2.Error as e:
        print(f"Error occurred while connecting to the database or executing query: {e}")

def get_last_data(player, conn):
    # SQL query to fetch the last 10 games for 'mp' and 'plus_minus' for a specified player
    query = f"""
    SELECT mp, plus_minus
    FROM nba  
    WHERE player = '{player}'
    ORDER BY date DESC 
    LIMIT 10
    """
    # Execute the query and read the data into a DataFrame
    df = pd.read_sql(query, conn)

    # Calculate the average of 'mp' and 'plus_minus'
    avg_mp = df['mp'].mean()
    avg_plus_minus = df['plus_minus'].mean()

    return avg_mp, avg_plus_minus

def get_soft_predictions(team, opp, player_df):
    # Load injured players from JSON file
    with open('injury.json', 'r') as file:
        injuries = json.load(file)

    # Filter the players by team and opponent
    team_players = player_df[player_df['team'] == team]['player'].tolist()
    opp_players = player_df[player_df['team'] == opp]['player'].tolist()

    # Initialize dictionaries to hold individual player data
    team_stats = {
        'pts': {}, 'trb': {}, 'ast': {}, 
        'p_r': {}, 'p_a': {}, 'a_r': {}, 'p_r_a': {}, 'blk': {}, 'tov': {}
    }
    opp_stats = {
        'pts': {}, 'trb': {}, 'ast': {}, 
        'p_r': {}, 'p_a': {}, 'a_r': {}, 'p_r_a': {}, 'blk': {}, 'tov': {}
    }

    # Function to populate player stats
    def populate_player_stats(players, stats, team_of_player):
        count = 0
        player_list = []
        for player in players:
            if player in injuries:  # If player is injured, set all their predicted stats to 0
                for key in stats:
                    stats[key][player] = 0
            else:

                for key in stats:
                    # Assuming soft function returns a predicted value or NaN for each stat
                    predicted_value = soft(player, opp if team_of_player == team else team, key, 1)
                    # Check if the predicted value is NaN and if so, use the average market value
                    if pd.isna(predicted_value):
                        count = count +1
                        player_list.append(player)
                        predicted_value = player_df.loc[player_df['player'] == player, f"avg_{key}"].values[0]
                    stats[key][player] = predicted_value
        print(set(player_list))
        print(count)

    # Populate stats for both teams
    populate_player_stats(team_players, team_stats, team)
    populate_player_stats(opp_players, opp_stats, opp)

    # Function to aggregate stats by player position
    def aggregate_position_data(stat_dict, df):
        position_data = {'G': 0, 'F': 0, 'C': 0}  # Initialize all positions with zero
        for player, stat in stat_dict.items():
            position = df.loc[df['player'] == player, 'pos'].values
            if position.size > 0:
                position_data[position[0]] += stat
            else:
                continue  # Skip if position is not determined

        # Calculate average only if there is data
        for pos in position_data.keys():
            count = sum(1 for player in stat_dict.keys() if df.loc[df['player'] == player, 'pos'].values == pos)
            if count > 0:
                position_data[pos] /= count

        return position_data

    # Prepare the final DataFrame
    results = {
        'team': [team],
        'opp': [opp],
        'teammates_points': [aggregate_position_data(team_stats['pts'], player_df)],
        'teammates_rebounds': [aggregate_position_data(team_stats['trb'], player_df)],
        'teammates_assists': [aggregate_position_data(team_stats['ast'], player_df)],
        'opponents_points': [aggregate_position_data(opp_stats['pts'], player_df)],
        'opponents_rebounds': [aggregate_position_data(opp_stats['trb'], player_df)],
        'opponents_assists': [aggregate_position_data(opp_stats['ast'], player_df)],
        'teammates_pr': [aggregate_position_data(team_stats['p_r'], player_df)],
        'teammates_pa': [aggregate_position_data(team_stats['p_a'], player_df)],
        'teammates_ar': [aggregate_position_data(team_stats['a_r'], player_df)],
        'opponents_pr': [aggregate_position_data(opp_stats['p_r'], player_df)],
        'opponents_pa': [aggregate_position_data(opp_stats['p_a'], player_df)],
        'opponents_ar': [aggregate_position_data(opp_stats['a_r'], player_df)],
        'teammates_pra': [aggregate_position_data(team_stats['p_r_a'], player_df)],
        'opponents_pra': [aggregate_position_data(opp_stats['p_r_a'], player_df)],
        'teammates_blocks': [aggregate_position_data(team_stats['blk'], player_df)],
        'teammates_turnovers': [aggregate_position_data(team_stats['tov'], player_df)],
        'opponents_blocks': [aggregate_position_data(opp_stats['blk'], player_df)],
        'opponents_turnovers': [aggregate_position_data(opp_stats['tov'], player_df)]

    }

    df = pd.DataFrame(results)

    # Expand each dictionary into separate columns and drop the original column
    for field in ['teammates_points', 'teammates_rebounds', 'teammates_assists',
                  'opponents_points', 'opponents_rebounds', 'opponents_assists',
                  'teammates_pr', 'teammates_pa', 'teammates_ar', 'opponents_pr',
                  'opponents_pa', 'opponents_ar', 'teammates_pra', 'opponents_pra',
                  'teammates_blocks', 'teammates_turnovers', 'opponents_blocks', 
                  'opponents_turnovers']:
        df_field = pd.json_normalize(df[field].iloc[0])
        df_field.columns = [f"{field}_{col}" for col in df_field.columns]  # Rename columns to include stat field
        df = pd.concat([df, df_field], axis=1)
        df.drop(field, axis=1, inplace=True)  # Drop the original column

    return df

def preprocess_data(df, market, transformers):
    features = ['plus_minus','opp','hoa','mp','teammates_points_G','teammates_points_F', 'teammates_points_C', 'teammates_rebounds_G',
        'teammates_rebounds_F', 'teammates_rebounds_C', 'teammates_assists_G',
        'teammates_assists_F', 'teammates_assists_C', 'opponents_points_G',
        'opponents_points_F', 'opponents_points_C', 'opponents_rebounds_G',
        'opponents_rebounds_F', 'opponents_rebounds_C', 'opponents_assists_G',
        'opponents_assists_F', 'opponents_assists_C', 'teammates_pr_G',
        'teammates_pr_F', 'teammates_pr_C', 'teammates_pa_G', 'teammates_pa_F',
        'teammates_pa_C', 'teammates_ar_G', 'teammates_ar_F', 'teammates_ar_C',
        'opponents_pr_G', 'opponents_pr_F', 'opponents_pr_C', 'opponents_pa_G',
        'opponents_pa_F', 'opponents_pa_C', 'opponents_ar_G', 'opponents_ar_F',
        'opponents_ar_C', 'teammates_pra_G', 'teammates_pra_F',
        'teammates_pra_C', 'opponents_pra_G', 'opponents_pra_F',
        'opponents_pra_C','teammates_blocks_F', 'teammates_blocks_C',
        'teammates_blocks_G','teammates_turnovers_F','teammates_turnovers_C',
        'teammates_turnovers_G','opponents_blocks_F','opponents_blocks_C',
        'opponents_blocks_G','opponents_turnovers_F','opponents_turnovers_C',
        'opponents_turnovers_G']
    
    target = market
    X, y = df[features], df[target]
    # Convert categorical columns and normalize
    X_transformed = transformers.fit_transform(X)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor

def train_nn(model, X_train, y_train, X_test, y_test, epochs=20000, batch_size=35):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Adding L2 regularization
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.sqrt(criterion(output, target))
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            predictions = model(X_test)
            test_loss = torch.sqrt(criterion(predictions, y_test))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss {loss.item()} Test Loss {test_loss.item()}')

    return model

if __name__ == "__main__":
    conn = create_engine('postgresql+psycopg2://postgres:gwdb@localhost:5600/mnrj')
    player = "Jayson Tatum"
    market = 'ast'
    nba_data = load_nba(player,conn)
    game_stats = load_game_stats(player,conn)
    df = nba_data.merge(
        game_stats,
        on= ["team", "opp","date"]
    )

    transformers = ColumnTransformer(transformers = [
        ('num', StandardScaler(), ['plus_minus','mp','hoa','teammates_points_G','teammates_points_F', 'teammates_points_C', 'teammates_rebounds_G',
        'teammates_rebounds_F', 'teammates_rebounds_C', 'teammates_assists_G',
        'teammates_assists_F', 'teammates_assists_C', 'opponents_points_G',
        'opponents_points_F', 'opponents_points_C', 'opponents_rebounds_G',
        'opponents_rebounds_F', 'opponents_rebounds_C', 'opponents_assists_G',
        'opponents_assists_F', 'opponents_assists_C', 'teammates_pr_G',
        'teammates_pr_F', 'teammates_pr_C', 'teammates_pa_G', 'teammates_pa_F',
        'teammates_pa_C', 'teammates_ar_G', 'teammates_ar_F', 'teammates_ar_C',
        'opponents_pr_G', 'opponents_pr_F', 'opponents_pr_C', 'opponents_pa_G',
        'opponents_pa_F', 'opponents_pa_C', 'opponents_ar_G', 'opponents_ar_F',
        'opponents_ar_C', 'teammates_pra_G', 'teammates_pra_F',
        'teammates_pra_C', 'opponents_pra_G', 'opponents_pra_F',
        'opponents_pra_C', 'teammates_blocks_F', 'teammates_blocks_C','teammates_blocks_G',
        'teammates_turnovers_F','teammates_turnovers_C','teammates_turnovers_G',
        'opponents_blocks_F','opponents_blocks_C','opponents_blocks_G',
        'opponents_turnovers_F','opponents_turnovers_C','opponents_turnovers_G']),  # Example features
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['opp'])
    ], remainder='passthrough')


    X, y = preprocess_data(df, market, transformers)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
    model = FeedforwardNN(X_train.shape[1])  # Create the model with the correct input size
    trained_model = train_nn(model, X_train, y_train, X_test, y_test)

    print("Model trained successfully!")