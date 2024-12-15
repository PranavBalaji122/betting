import pandas as pd
import psycopg2
from psycopg2 import sql

#remove the column 'Rk' as it will be replaced with an incrementing 'id'
df = pd.read_csv('data.csv')
df = df.drop(['Rk'], axis=1)
df.to_csv('data.csv', index=False)

conn = psycopg2.connect(
    host = "localhost", 
    dbname = "mnrj", 
    user= "postgres", 
    password = "gwdb", 
    port = 5600
)
cursor = conn.cursor()

# Load data from the cleaned CSV file
with open('data.csv', 'r') as f:
    next(f)  # Skip the header row
    cursor.copy_from(f, 'public.nba', sep=',', null='')  # Assuming empty strings for null values

# Apply transformations and updates
cursor.execute("UPDATE public.nba SET result = REPLACE(result, ' (OT)', '');")
cursor.execute("UPDATE public.nba SET resultChar = TRIM(SUBSTRING(result FROM '^[WL]')), score1 = CAST(TRIM(SPLIT_PART(SUBSTRING(result FROM '[0-9]+-[0-9]+'), '-', 1)) AS INTEGER), score2 = CAST(TRIM(SPLIT_PART(SUBSTRING(result FROM '[0-9]+-[0-9]+'), '-', 2)) AS INTEGER);")
cursor.execute("UPDATE public.nba SET total_score = score1 + score2;")
cursor.execute("ALTER TABLE public.nba DROP COLUMN score1, DROP COLUMN score2;")
cursor.execute("UPDATE public.nba SET age = CAST(SUBSTRING(age FROM 1 FOR 2) AS INTEGER);")
cursor.execute("ALTER TABLE public.nba ALTER COLUMN age TYPE INTEGER USING CAST(age AS INTEGER);")
cursor.execute("UPDATE public.nba SET hoa = CASE WHEN hoa IS NULL THEN '0' WHEN hoa = '@' THEN '1' END;")
cursor.execute("ALTER TABLE public.nba ALTER COLUMN hoa TYPE INTEGER USING CAST(hoa AS INTEGER);")
cursor.execute("UPDATE public.nba SET gs = CASE WHEN gs IS NULL THEN '0' WHEN gs = '*' THEN '1' END;")
cursor.execute("ALTER TABLE public.nba ALTER COLUMN gs TYPE INTEGER USING CAST(gs AS INTEGER);")
cursor.execute("DELETE FROM public.nba WHERE mp = 0;")
cursor.execute("UPDATE public.nba SET fg_percent = COALESCE(fg_percent, 0), twop_percent = COALESCE(twop_percent, 0);")
cursor.execute("DELETE FROM public.nba WHERE fg_percent = 0;")
cursor.execute("UPDATE public.nba SET month = EXTRACT(MONTH FROM date);")
cursor.execute("ALTER TABLE public.nba RENAME COLUMN resultChar TO result;")
cursor.execute("UPDATE public.nba SET result = CASE WHEN result = 'L' THEN 0 WHEN result = 'W' THEN 1 END;")
cursor.execute("ALTER TABLE public.nba ALTER COLUMN result TYPE INTEGER USING CAST(result AS INTEGER);")

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()
