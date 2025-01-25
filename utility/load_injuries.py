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

    # Filter the DataFrame to include only rows where the status is 'Out'
    df_filtered = df[df['status'] == 'Out']

    # Select only the 'player' and 'team' columns
    df_final = df_filtered[['player', 'team']]

    # Create a dictionary grouping players by team
    team_grouped = df_final.groupby('team')['player'].apply(list).to_dict()

    # Serialize the dictionary to JSON and write to a file
    with open('JSON/injury.json', 'w') as file:
        json.dump(team_grouped, file, indent=2)

if __name__ == '__main__':
    load_injuries()