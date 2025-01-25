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
        # cursor.execute("""
        # SELECT json_agg(json_build_object('player', player, 'position', pos)) FROM (
        #     SELECT player, pos, AVG(mp) as avg_minutes
        #     FROM public.nba
        #     WHERE team = %s AND player != %s
        #     GROUP BY player, pos
        #     ORDER BY avg_minutes DESC
        #     LIMIT 8
        # ) as top_teammates;
        # """, (player_team, description))
        # teammates = cursor.fetchone()[0]

        # # Query for top 8 opposing players based on average minutes played
        # cursor.execute("""
        # SELECT json_agg(json_build_object('player', player, 'position', pos)) FROM (
        #     SELECT player, pos, AVG(mp) as avg_minutes
        #     FROM public.nba
        #     WHERE team = %s
        #     GROUP BY player, pos
        #     ORDER BY avg_minutes DESC
        #     LIMIT 8
        # ) as top_opponents;
        # """, (opp_team,))
        # opposing_players = cursor.fetchone()[0]

        return {
            "player": description,
            # "position": player_position,
            "team": player_team,
            # "teammates": teammates,
            "opp": opp_team,
            # "opposing_players": opposing_players,
            "hoa": hoa
        }
    else:
        return None

def process_props_and_output(cursor, data):
    """
    Processes betting props for multiple platforms and markets, returning detailed data.
    """
    market_mapping = {
        "player_points": "pts",
        "player_rebounds": "trb",
        "player_assists": "ast",
        "player_points_rebounds_assists": "p_r_a"
    }
    results = {}
    for platform, markets in data.items():
        platform_results = []
        for market, props in markets.items():
            processed_props = {}
            feature_column = market_mapping.get(market)
            if not feature_column:
                continue  # Skip if market is not mapped
            for prop in props:
                key = (prop["description"], prop["game_id"])
                if key not in processed_props:
                    player_info = fetch_player_details(cursor, prop["description"], prop["home_team"], prop["away_team"])
                    if player_info:
                        player_info.update({
                            "line": prop["point"],
                            "market": feature_column,
                            prop["name"].lower(): prop["price"]  # dynamically add "over" or "under"
                        })
                        processed_props[key] = player_info
                else:
                    processed_props[key].update({
                        prop["name"].lower(): prop["price"]
                    })
            platform_results.extend(processed_props.values())
        results[platform] = platform_results
    return results

def write_json_file(data, file_path):
    """
    Writes the given data to a JSON file without escaping non-ASCII characters.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
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
        data = read_json_file('JSON/props.json')
        detailed_player_props = process_props_and_output(cursor, data)
        write_json_file(detailed_player_props, 'JSON/processed_odds.json')
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
