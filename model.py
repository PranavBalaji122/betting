from io import StringIO
import json
import psycopg2
from psycopg2.extras import Json
from psycopg2 import sql
from sklearn import datasets
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def load_nba(player):
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="mnrj",
            user="postgres",
            password="gwdb",
            port=5600
        )
        # Use the connection to execute the query
        query = f"SELECT * FROM nba WHERE player = '{player}';"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except psycopg2.Error as e:
        print(f"Error occurred while connecting to the database or executing query: {e}")
        return None

def load_player_positions():
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="mnrj",
            user="postgres",
            password="gwdb",
            port=5600
        )
        # Use the connection to execute the query
        query = f"SELECT * FROM latest_player_teams;"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except psycopg2.Error as e:
        print(f"Error occurred while connecting to the database or executing query: {e}")
        return None

def load_game_stats(player):
    positions_df = load_player_positions()
    if positions_df is None:
        return pd.DataFrame()

    try:
        conn = psycopg2.connect(host="localhost", dbname="mnrj", user="postgres", password="gwdb", port=5600)
        query = f"""
        SELECT *
        FROM game_stats
        WHERE teammates_points::jsonb ? '{player}';  -- Checking if player key exists in JSON
        """
        df = pd.read_sql(query, conn)
        conn.close()
        def aggregate_position_data(data, exclude_player):
            position_totals = {'G': 0, 'F': 0, 'C': 0}
            total_contribution = 0  # To hold the sum of all contributions for normalization
            
            # Sum values by position
            for player, value in data.items():
                pos = positions_df.loc[positions_df['player'] == player, 'pos'].values
                if pos:
                    position_totals[pos[0]] += value
                    total_contribution += value
            
            # Subtract the contribution of the excluded player after the total is calculated
            if exclude_player in data:
                player_pos = positions_df.loc[positions_df['player'] == exclude_player, 'pos'].values
                if player_pos:
                    position_totals[player_pos[0]] -= data[exclude_player]
                    total_contribution -= data[exclude_player]
            
            # Convert totals to percentages
            if total_contribution > 0:  # Avoid division by zero
                for position in position_totals:
                    position_totals[position] = round(position_totals[position] / total_contribution, 2)  # Round to 2 decimal places

            return position_totals

        # Aggregate data based on positions, removing the current player
        def aggregate_position_data(data, exclude_player):
            if exclude_player in data:
                del data[exclude_player]  # Remove the player from the data
            
            position_totals = {'G': 0, 'F': 0, 'C': 0}
            total_contribution = 0  # To hold the sum of all contributions for normalization

            # Sum values by position
            for player, value in data.items():
                pos = positions_df.loc[positions_df['player'] == player, 'pos'].values
                if pos:
                    position_totals[pos[0]] += value
                    total_contribution += value

            # Convert totals to percentages
            if total_contribution > 0:  # Avoid division by zero
                for position in position_totals:
                    position_totals[position] = round(position_totals[position] / total_contribution, 2)  # Round to 2 decimal places

            return position_totals
            
            return position_totals
        stats_fields = ['teammates_points', 'teammates_rebounds', 'teammates_assists', 'opponents_points', 'opponents_rebounds', 'opponents_assists',
                        'teammates_pr','teammates_pa','teammates_ar','opponents_pr','opponents_pa','opponents_ar','teammates_pra','opponents_pra']
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


# Function to train the random forest model
def train_random_forest(player):
    # Drop unnecessary columns
    nba_data = load_nba(player)
    game_stats = load_game_stats(player)
    data = nba_data.merge(
        game_stats,
        on= ["team", "opp","date"]
    )
    print(data)
    if(data.empty):
        return 0
    return data



def run(player, opponent, hoa, market):
    print(train_random_forest(player).columns)



run("Jayson Tatum", "CHI", 1, "pts")


columns_to_keep = [
        'hoa','opp','teammates_points',
            "teammates_rebounds",
            "teammates_assists",
            "teammates_pr",
            "teammates_pa",
            "teammates_ar",
            "teammates_pra",
            "opponents_points",
            "opponents_rebounds",
            "opponents_assists",
            "opponents_pr",
            "opponents_pa",
            "opponents_ar",
            "opponents_pra"]