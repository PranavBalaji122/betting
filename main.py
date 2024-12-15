
import psycopg2
import pandas as pd

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host = "localhost", 
    dbname = "mnrj", 
    user= "postgres", 
    password = "gwdb", 
    port = 5600
)

# Query the data
query = "SELECT * FROM nba;"
data = pd.read_sql(query, conn)

# Close the connection
conn.close()

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical columns (if necessary)
label_encoders = {}  # Store encoders for inverse transformation later
for col in data.select_dtypes(include=['object']):
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Standardize numeric features
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data)







