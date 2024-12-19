from sklearn import datasets
from sklearn.model_selection import train_test_split
import psycopg2
import pandas as pd
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from calculateNewFeatures import calulateNewFeatures
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



def getPrediction():
    df = load_data()
    player = 'Jayson Tatum'
    opp = 'MIA'     
    features = calulateNewFeatures(df, player, opp)

def main():
    getPrediction()

if __name__ == '__main__':
    main()







