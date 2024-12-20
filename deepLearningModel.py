import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import psycopg2
import json
from sklearn.preprocessing import LabelEncoder





def get_data_for_training():
    conn = psycopg2.connect(
        host="localhost",
        dbname="mnrj",
        user="postgres",
        password="gwdb",
        port=5600
    )
    df = pd.read_sql(f"select * from game_stats", conn)
    conn.close()
    df.to_csv('sql.csv', encoding='utf-8', index=False)
    return df

def parseData(df, player_name):
    df['teammates_points'] = df['teammates_points']
    df['teammates_rebounds'] = df['teammates_rebounds']
    df['teammates_assists'] = df['teammates_assists']
    df['opponents_points'] = df['opponents_points']
    df['opponents_rebounds'] = df['opponents_rebounds']
    df['opponents_assists'] = df['opponents_assists']
    df['opponent'] = df['opponent']

    label_encoder = LabelEncoder()

    df['opponent_encoded'] = label_encoder.fit_transform(df['opponent'])
    print(df[['opponent', 'opponent_encoded']].drop_duplicates())

    
    X, y = [], []
    for index, row in df.iterrows():
        player_stats = row['teammates_points'].pop(player_name, None)
        if player_stats is not None:
            y.append(player_stats)  # Use player's points as the label
            features = []
            # Iterate over the specific columns, checking their data type before appending
            for stat in ['teammates_points', 'teammates_rebounds', 'teammates_assists',
                         'opponents_points', 'opponents_rebounds', 'opponents_assists']:
                if isinstance(row[stat], dict):  # Check if the item is a dictionary
                    features.extend(row[stat].values())
                else:
                    features.append(row[stat])  # Append directly if it's numeric or another non-dictionary type
            # Append 'opponent_encoded' directly as it is an integer
            features.append(row['opponent_encoded'])
            X.append(features)

    return np.array(X), np.array(y)


def trainModel(playerName):
    data = get_data_for_training()
    X, y = parseData(data, playerName)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Define the input layer
            self.fc1 = nn.Linear(X_train.shape[1], 100)

            # Create a dictionary to hold the intermediate layers
            self.layers = nn.ModuleDict({
                f'fc{i}': nn.Linear(100, 100) for i in range(2, 40)  # 38 additional layers
            })

            # Define the output layer
            self.output = nn.Linear(100, 1)

        def forward(self, x):
            # Pass through the first layer
            x = torch.relu(self.fc1(x))

            # Pass through the intermediate layers
            for i in range(2, 40):
                layer = self.layers[f'fc{i}']
                x = torch.relu(layer(x))

            # Pass through the output layer
            x = self.output(x)
            return x
    
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

     # Convert arrays to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Train the model
    model.train()
    for epoch in range(1000):  # Number of epochs
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs.squeeze(), y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        predicted = model(X_test_t).squeeze()
        mse = criterion(predicted, y_test_t)
        print(f'Test MSE: {mse.item()}')
    data = [2, 0, 19, 6, 16, 14, 20, 5, 0, 3, 1, 8, 20, 8, 2, 0, 4, 4, 4, 3, 0, 12, 10, 9, 10, 10, 5, 12, 5, 5, 9, 0, 3, 7, 8, 1, 1, 4, 6, 0, 2, 0, 3, 0, 3, 1, 2, 2]
    print(len(data))
    data = np.array([data])
    data_tensor = torch.FloatTensor(data)
    model.eval() 
    with torch.no_grad():  # Context-manager that disables gradient calculation (for inference)
        prediction = model(data_tensor)
        print("Predicted output:", prediction)

    




trainModel("Jayson Tatum")

#gatherData("LeBron James", "LAL", "GSW", "pts")

# dataset = TensorDataset(X, y)
# train_loader = DataLoader(dataset, batch_size=10, shuffle=True)