import requests
import pandas as pd
import json

def load_injuries():
    # Replace 'api_endpoint_url' with the actual URL of the API endpoint
    api_endpoint_url = "https://www.rotowire.com/basketball/tables/injury-report.php?team=ALL&pos=ALL"
    response = requests.get(api_endpoint_url)

    # Convert the JSON response to a Python dictionary
    data = response.json()

    # Create a DataFrame from the response data
    df = pd.DataFrame(data)
    
    # Select the columns we care about: player, team, and status
    df_final = df[['player', 'team', 'status']]

    # Group by 'team' and collect each row as a dictionary of {player, status}
    # This will result in a structure like:
    # {
    #   "ATL": [
    #       {"player": "Player A", "status": "Out"},
    #       {"player": "Player B", "status": "Day-To-Day"}
    #       ...
    #   ],
    #   "BOS": [
    #       ...
    #   ],
    #   ...
    # }
    team_grouped = (
        df_final
        .groupby('team')
        .apply(lambda group: group[['player', 'status']].to_dict(orient='records'))
        .to_dict()
    )

    # Serialize the dictionary to JSON and write to a file
    with open('JSON/injury.json', 'w') as file:
        json.dump(team_grouped, file, indent=2)

if __name__ == '__main__':
    load_injuries()
