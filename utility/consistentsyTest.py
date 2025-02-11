import psycopg2
import pandas as pd
import os

def getConsistency(feature):
    try:
        conn = psycopg2.connect(
            host = os.getenv("DB_HOST"), 
            dbname = os.getenv("DB_NAME"), 
            user= os.getenv("DB_USER"), 
            password = os.getenv("DB_PASS"), 
            port = os.getenv("DB_PORT")
        )
        query = f"""
                WITH RecentGames AS (
                    SELECT
                        player,
                        team,
                        {feature},
                        mp,
                        ROW_NUMBER() OVER (PARTITION BY player ORDER BY days_since DESC) AS rn
                    FROM
                        nba
                )
                SELECT
                    player,
                    team,
                    AVG({feature}) AS average_{feature},
                    STDDEV({feature}) AS stddev_{feature},
                    CASE 
                        WHEN AVG({feature}) = 0 THEN NULL
                        ELSE (STDDEV({feature}) / AVG({feature}))
                    END AS cv_{feature}
                FROM
                    RecentGames
                WHERE
                    rn <= 20
                GROUP BY
                    player, team
                HAVING
                    AVG({feature}) > 0 AND
                    AVG(mp) > 10  -- Ensuring the average minutes played over the last 15 games is more than 10
                ORDER BY
                    cv_{feature} ASC
                FETCH NEXT 100 ROWS ONLY;
        """
        player_data = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

    playerNames = player_data['player'].tolist()
    teams = player_data['team'].tolist()
    return playerNames, teams

def main():
    print(getConsistency('p_r'))

if __name__ == '__main__':
    main()
