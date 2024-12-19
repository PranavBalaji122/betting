import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from calculateNewFeatures import calulateNewFeatures


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
def train_random_forest_model(data, player):
    # Drop unnecessary columns
    data = data[data['player'] == player]
    columns_to_keep = [
    'result', 'total_score', 'mp', 'fga', 'fg_percent', 'twop', 
    'twop_percent', 'threep', 'ft', 'ft_percent', 'ts_percent', 
    'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'gmsc', 'pts'
    ]
    data = data.loc[:, columns_to_keep]

    # Drop rows with missing values
    data = data.dropna()
    
    

    # Separate features and target variable
    X = data.drop(['pts'], axis=1)  # Assuming 'pts' is the target column
    y = data['pts']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Evaluate the model (optional)
    predictions = random_forest_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Return the trained model
    return random_forest_model


# Function to predict points for a specific player and opponent
def predict_points(player, opp, model, df):
    features = calulateNewFeatures(df, player, opp)
    features = features.drop(columns=['pts'])
    prediction = model.predict(features)

    print(f"Predicted points for {player} against {opp}: {prediction[0]}")
    return prediction[0]


# Main function to train the model and make a prediction
def main():
    player = 'Isaiah Joe'
    opp = 'ORL'

    df = load_data()
    random_forest_model = train_random_forest_model(df, player)

    # Make a prediction for a specific player and opponent    
    predict_points(player, opp, random_forest_model, df)


if __name__ == '__main__':
    main()
