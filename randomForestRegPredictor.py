import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from calculateNewFeatures import calulateNewFeatures
from sklearn.preprocessing import LabelEncoder


# Function to load data
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


# Function to train the random forest model
def train_random_forest_model(data, player, target):
    # Drop unnecessary columns
    data = data[data['player'] == player]

    columns_to_keep = [
        'total_score', 'mp', 'fga', 'fg_percent', 'twop', 
        'twop_percent', 'threep', 'ft', 'ft_percent', 'ts_percent', 
        'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'gmsc','pts'
    ]
    data = data.loc[:, columns_to_keep]
    
    # Drop rows with missing values

    data = data.dropna()



    # Separate features and target variable
    X = data.drop([target], axis=1)  # target is the target column
    y = data[target]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    random_forest_model = RandomForestRegressor(n_estimators=10000, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Evaluate the model (optional)
    predictions = random_forest_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Return the trained model
    return random_forest_model


# Function to predict points for a specific player and opponent
def predict_points(player, opp, model, target):
    features = calulateNewFeatures(player, opp)
    features = features.drop(columns=[target])
    prediction = model.predict(features)

    print(f"Predicted {target} for {player} against {opp}: {prediction[0]}")
    return prediction[0]


# Main function to train the model and make a prediction
def main(player, opp, target):
    pridectedPoints = 0

    df = load_data()
    random_forest_model = train_random_forest_model(df, player, target)

    # Make a prediction for a specific player and opponent    
    pridectedPoints = predict_points(player, opp, random_forest_model, target)
    return pridectedPoints

def run(player, opp, target):
    pridectedPoints = 0
    pridectedPoints = main(player, opp, target)
    return pridectedPoints

if __name__ == '__main__':
    player = 'Jayson Tatum'
    opponent = 'CHI'
    target = 'pts'
    main(player, opponent, target)
