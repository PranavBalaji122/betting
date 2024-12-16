import json
import psycopg2

def read_json_file(file_path):
    """
    Reads a JSON file using UTF-8 encoding and returns the data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def fetch_player_details(cursor, description, home_team, away_team):
    """
    Fetches the most recent team information for the player, details of top 8 teammates by average minutes played
    (excluding the player themselves), and top 8 opposing players by average minutes played.
    """
    cursor.execute("""
    SELECT team, pos FROM public.latest_player_teams WHERE player = %s;
    """, (description,))
    result = cursor.fetchone()
    if result:
        player_team, player_position = result
        hoa = "home" if player_team == home_team else "away"
        opp_team = home_team if player_team == away_team else away_team

        # Query for top 8 teammates based on average minutes played, excluding the player
        cursor.execute("""
        SELECT json_agg(json_build_object('player', player, 'position', pos)) FROM (
            SELECT player, pos, AVG(mp) as avg_minutes
            FROM public.nba
            WHERE team = %s AND player != %s
            GROUP BY player, pos
            ORDER BY avg_minutes DESC
            LIMIT 8
        ) as top_teammates;
        """, (player_team, description))
        teammates = cursor.fetchone()[0]

        # Query for top 8 opposing players based on average minutes played
        cursor.execute("""
        SELECT json_agg(json_build_object('player', player, 'position', pos)) FROM (
            SELECT player, pos, AVG(mp) as avg_minutes
            FROM public.nba
            WHERE team = %s
            GROUP BY player, pos
            ORDER BY avg_minutes DESC
            LIMIT 8
        ) as top_opponents;
        """, (opp_team,))
        opposing_players = cursor.fetchone()[0]

        return {
            "player": description,
            "position": player_position,
            "team": player_team,
            "teammates": teammates,
            "opp": opp_team,
            "opposing_players": opposing_players,
            "hoa": hoa
        }
    else:
        return None





def process_props_and_output(cursor, data):
    """
    Processes betting props for multiple platforms and markets, returning detailed data.
    """
    results = {}
    for platform, markets in data.items():
        platform_results = []
        for market, props in markets.items():
            for prop in props:
                player_info = fetch_player_details(cursor, prop["description"], prop["home_team"], prop["away_team"])
                if player_info:
                    player_info.update({
                        "line": prop["point"],
                        "odd": prop["price"],
                        "market": market  # Include the market in the details
                    })
                    platform_results.append(player_info)
        results[platform] = platform_results
    return results

def write_json_file(data, file_path):
    """
    Writes the given data to a JSON file without escaping non-ASCII characters.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # ensure_ascii=False prevents Unicode escaping
    print(f"Data written to {file_path} successfully.")


def main():
    """
    Main function to execute the whole process.
    """
    conn = psycopg2.connect(
        host = "localhost", 
        dbname = "mnrj", 
        user= "postgres", 
        password = "gwdb", 
        port = 5600
    )
    cursor = conn.cursor()

    try:
        data = read_json_file('props.json')
        detailed_player_props = process_props_and_output(cursor, data)
        write_json_file(detailed_player_props, 'processed_odds.json')

        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
