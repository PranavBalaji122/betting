from io import StringIO
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from psycopg2 import sql

def process_csv():
    csv_columns = [
        'Player', 'Date', 'Age', 'Team', 'HOA', 'Opp', 'Result', 'GS', 'MP', 'FG', 'FGA',
        'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TS%', 'ORB', 
        'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'BPM', '+/-', 'Pos.', 
        'Player-additional'
    ]

    # Corresponding SQL column names (excluding 'id' which is auto-generated)
    sql_columns = [
        'player', 'date', 'age', 'team', 'hoa', 'opp', 'result', 'gs', 'mp', 'fg', 'fga',
        'fg_percent', 'twop', 'twopa', 'twop_percent', 'threep', 'threepa', 'threep_percent',
        'ft', 'fta', 'ft_percent', 'ts_percent', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 
        'tov', 'pf', 'pts', 'gmsc', 'bpm', 'plus_minus', 'pos', 'player_additional'
    ]

    column_mapping = dict(zip(csv_columns, sql_columns))
    df = pd.read_csv('data.csv', keep_default_na=False)
    # df = df.drop(['Rk'], axis=1)
    df.rename(columns=column_mapping, inplace=True)

    # Numeric columns that may contain empty strings
    numeric_columns = ['fg_percent', 'twop_percent', 'threep_percent', 'ft_percent', 'ts_percent','plus_minus']
    for column in numeric_columns:
        df[column] = df[column].apply(lambda x: "None" if x.strip() == "" else float(x))

    df.to_csv('modified_data.csv', index=False)
    print("CSV has been rewritten with SQL-compatible column names and corrected numeric fields.")


def create_table(cursor):
    cursor.execute("""
        CREATE TABLE public.nba
        (
            id SERIAL PRIMARY KEY,
            player VARCHAR(50),
            date DATE,
            age VARCHAR(10),
            team VARCHAR(10),
            hoa CHAR(1),
            opp VARCHAR(10),
            result VARCHAR(20),
            gs CHAR(1),
            mp INTEGER,
            fg INTEGER,
            fga INTEGER,
            fg_percent NUMERIC(5,3) NULL,  -- Allow NULLs
            twop INTEGER,
            twopa INTEGER,
            twop_percent NUMERIC(5,3) NULL,  -- Allow NULLs
            threep INTEGER,
            threepa INTEGER,
            threep_percent NUMERIC(5,3) NULL,  -- Allow NULLs
            ft INTEGER,
            fta INTEGER,
            ft_percent NUMERIC(5,3) NULL,  -- Allow NULLs
            ts_percent NUMERIC(5,3) NULL,  -- Allow NULLs
            orb INTEGER,
            drb INTEGER,
            trb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            pts INTEGER,
            gmsc NUMERIC(5,1),
            bpm NUMERIC(5,1),
            plus_minus NUMERIC(5,1),
            pos VARCHAR(4),
            player_additional VARCHAR(20)
        );
    """)

def load_data(cursor):
    with open('modified_data.csv', 'r') as f:
        next(f)  # This skips the header line to prevent it from being read as data
        cursor.copy_from(f, 'nba', sep=',', null='None', columns=('player', 'date', 'age', 'team', 'hoa', 'opp', 'result', 'gs', 'mp', 'fg', 'fga', 'fg_percent', 'twop', 'twopa', 'twop_percent', 'threep', 'threepa', 'threep_percent', 'ft', 'fta', 'ft_percent', 'ts_percent', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc', 'bpm', 'plus_minus', 'pos', 'player_additional'))


def perform_updates(cursor):
    # List of SQL updates and modifications to perform
    updates = [
        "UPDATE public.nba SET result = REPLACE(result, ' (OT)', '');",
        "ALTER TABLE public.nba ADD COLUMN resultChar CHAR(1), ADD COLUMN score1 INTEGER, ADD COLUMN score2 INTEGER;",
        "UPDATE public.nba SET resultChar = TRIM(SUBSTRING(result FROM '^[WL]')), score1 = CAST(TRIM(SPLIT_PART(SUBSTRING(result FROM '[0-9]+-[0-9]+'), '-', 1)) AS INTEGER), score2 = CAST(TRIM(SPLIT_PART(SUBSTRING(result FROM '[0-9]+-[0-9]+'), '-', 2)) AS INTEGER);",
        "ALTER TABLE public.nba ADD COLUMN total_score INTEGER;",
        "UPDATE public.nba SET total_score = score1 + score2;",
        "ALTER TABLE public.nba DROP COLUMN result, DROP COLUMN score1, DROP COLUMN score2;",
        "ALTER TABLE public.nba ADD COLUMN p_r_a INTEGER GENERATED ALWAYS AS (pts + trb + ast) STORED, ADD COLUMN p_r INTEGER GENERATED ALWAYS AS (pts + trb) STORED, ADD COLUMN p_a INTEGER GENERATED ALWAYS AS (pts + ast) STORED, ADD COLUMN a_r INTEGER GENERATED ALWAYS AS (ast + trb) STORED;",
        "UPDATE public.nba SET age = CAST(SUBSTRING(age FROM 1 FOR 2) AS INTEGER);",
        "ALTER TABLE public.nba ALTER COLUMN age TYPE INTEGER USING CAST(age AS INTEGER);",
        "UPDATE public.nba SET hoa = CASE WHEN hoa = '' THEN '0' WHEN hoa = '@' THEN '1' END;",
        "ALTER TABLE public.nba ALTER COLUMN hoa TYPE INTEGER USING CAST(hoa AS INTEGER);",
        "UPDATE public.nba SET gs = CASE WHEN gs = '' THEN '0' WHEN gs = '*' THEN '1' END;",
        "ALTER TABLE public.nba ALTER COLUMN gs TYPE INTEGER USING CAST(gs AS INTEGER);",
        "DELETE FROM public.nba WHERE mp = 0;",
        "UPDATE public.nba SET fg_percent = COALESCE(fg_percent, 0), twop_percent = COALESCE(twop_percent, 0);",
        "DELETE FROM public.nba WHERE fg_percent = 0;",
        "ALTER TABLE public.nba ADD COLUMN month INTEGER;",
        "UPDATE public.nba SET month = EXTRACT(MONTH FROM date);",
        "ALTER TABLE public.nba ADD COLUMN days_since INTEGER;",
        "UPDATE public.nba SET days_since = date - DATE '2023-10-24';",
        "ALTER TABLE public.nba DROP COLUMN threepa, DROP COLUMN threep_percent, DROP COLUMN fta, DROP COLUMN twopa;",
        "ALTER TABLE public.nba RENAME COLUMN resultChar TO result;",
        "UPDATE public.nba SET result = CASE WHEN result = 'L' THEN 0 WHEN result = 'W' THEN 1 END;",
        "ALTER TABLE public.nba ALTER COLUMN result TYPE INTEGER USING CAST(result AS INTEGER);"
    ]

    for command in updates:
        cursor.execute(command)
        
def create_most_recent_player_team_table(cursor):
    sql_query = """
    WITH RankedPlayerTeams AS (
        SELECT
            player,
            team,
            pos,  -- Include the position column
            date,
            ROW_NUMBER() OVER (PARTITION BY player ORDER BY date DESC) AS rn
        FROM
            public.nba
    )
    SELECT
        player,
        team,
        pos  -- Include the position in the final SELECT
    INTO
        public.latest_player_teams
    FROM
    RankedPlayerTeams
    WHERE
        rn = 1;
    """
    cursor.execute(sql_query)
    print("Most recent player-team table created successfully.")


def update_game_details(cursor):
    cursor.execute("""
        ALTER TABLE public.nba
            ADD COLUMN IF NOT EXISTS teammates_rebounds JSONB,
            ADD COLUMN IF NOT EXISTS teammates_assists JSONB,
            ADD COLUMN IF NOT EXISTS teammates_pr JSONB,
            ADD COLUMN IF NOT EXISTS teammates_pa JSONB,
            ADD COLUMN IF NOT EXISTS teammates_ar JSONB,
            ADD COLUMN IF NOT EXISTS teammates_pra JSONB,
            ADD COLUMN IF NOT EXISTS opponents_rebounds JSONB,
            ADD COLUMN IF NOT EXISTS opponents_assists JSONB,
            ADD COLUMN IF NOT EXISTS opponents_pr JSONB,
            ADD COLUMN IF NOT EXISTS opponents_pa JSONB,
            ADD COLUMN IF NOT EXISTS opponents_ar JSONB,
            ADD COLUMN IF NOT EXISTS opponents_pra JSONB;
    """)
    cursor.execute("""
        SELECT id, player, date, team, opp FROM public.nba;
    """)
    games = cursor.fetchall()

    for game in games:
        game_id, player_name, game_date, team, opponent_team = game
        
        for relation, team_to_query in [('teammates', team), ('opponents', opponent_team)]:
            cursor.execute(f"""
                WITH RelevantPlayers AS (
                    SELECT
                        player,
                        trb,
                        ast,
                        pts,
                        RANK() OVER (ORDER BY mp DESC) AS rank
                    FROM
                        public.nba
                    WHERE
                        team = %s AND
                        date = %s AND
                        player != %s
                )
                SELECT
                    jsonb_agg(jsonb_build_object('player', player, 'points', pts)),
                    jsonb_agg(jsonb_build_object('player', player, 'rebounds', trb)),
                    jsonb_agg(jsonb_build_object('player', player, 'assists', ast)),
                    jsonb_agg(jsonb_build_object('player', player, 'pr', pts + trb)),
                    jsonb_agg(jsonb_build_object('player', player, 'pa', pts + ast)),
                    jsonb_agg(jsonb_build_object('player', player, 'ar', ast + trb)),
                    jsonb_agg(jsonb_build_object('player', player, 'pra', pts + trb + ast))
                FROM RelevantPlayers
                WHERE rank <= 8;
            """, (team_to_query, game_date, player_name))
            points, rebounds, assists, pr, pa, ar, pra = cursor.fetchone()

            # Update the respective columns based on whether it's teammates or opponents
            cursor.execute(f"""
                UPDATE public.nba
                SET 
                    {relation}_points = %s,
                    {relation}_rebounds = %s,
                    {relation}_assists = %s,
                    {relation}_pr = %s,
                    {relation}_pa = %s,
                    {relation}_ar = %s,
                    {relation}_pra = %s
                WHERE id = %s;
            """, (Json(points), Json(rebounds), Json(assists), Json(pr), Json(pa), Json(ar), Json(pra), game_id))



def main():
    conn = psycopg2.connect(
        host="localhost", 
        dbname="mnrj", 
        user="postgres", 
        password="gwdb", 
        port="5600"
    )
    cursor = conn.cursor()

    try:
        process_csv()
        create_table(cursor)
        load_data(cursor)
        perform_updates(cursor)
        create_most_recent_player_team_table(cursor)
        update_game_details(cursor)  # New function to update teammate details
        conn.commit()
        print("Pipeline executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
