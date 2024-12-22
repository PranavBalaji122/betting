import psycopg2
import pandas as pd


def getConsistency(feature):
    try:
        conn = psycopg2.connect(
            host="localhost", 
            dbname="mnrj", 
            user="postgres", 
            password="gwdb", 
            port="5600"
        )
        query = f"""
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
                    nba
                WHERE
                    mp > 15  -- Ensure that minutes played is greater than 15
                GROUP BY
                    player, team  -- Include 'team' in the GROUP BY clause
                HAVING AVG({feature}) > 0
                ORDER BY
                    cv_{feature} ASC
                OFFSET 7  -- Skip the first player to start from the second
                FETCH NEXT 50 ROWS ONLY;  -- Fetch the next 51 players (ranked 2 to 52)
        """
        player_data = pd.read_sql_query(query, conn)
        # print("Pipeline executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

    # print(player_data)
    playerNames = player_data['player'].tolist()
    teams = player_data['team'].tolist()
    return playerNames, teams



def main():
    print(getConsistency('ast'))

if __name__ == '__main__':
    main()
