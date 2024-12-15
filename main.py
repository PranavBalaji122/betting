
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
query = "SELECT * FROM nba ORDER BY days_since;"
data = pd.read_sql(query, conn)

print(data.head(10))

conn.close()







