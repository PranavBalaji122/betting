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
   
    label_encoder = LabelEncoder()
    df['opponent_encoded'] = label_encoder.fit_transform(df['opponent'])
    print(df[['opponent', 'opponent_encoded']].drop_duplicates())
    
    X, y = [], []
    for index, row in df.iterrows():
        player_stats = row['teammates_points'].pop(player_name, None)
        row['teammates_rebounds'].pop(player_name, None)
        row['teammates_assists'].pop(player_name, None)
        if player_stats is not None:
            y.append(player_stats)  
            features = []
            for stat in ['teammates_points', 'teammates_rebounds', 'teammates_assists',
                         'opponents_points', 'opponents_rebounds', 'opponents_assists']:
                            features.extend(row[stat].values())

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
            self.fc1 = nn.Linear(X_train.shape[1], 50)
            self.dropout1 = nn.Dropout(0.25)

            # Dynamically creating 200 hidden layers
            self.hidden_layers = nn.ModuleList()
            for _ in range(200):  # Create 200 layers of size 50
                self.hidden_layers.append(nn.Linear(50, 50))
                self.hidden_layers.append(nn.Dropout(0.25))  # Adding dropout to each layer

            self.fc_last = nn.Linear(50, 50)  # Final hidden layer before the output
            self.output = nn.Linear(50, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    x = torch.relu(layer(x))
                elif isinstance(layer, nn.Dropout):
                    x = layer(x)
            x = torch.relu(self.fc_last(x))
            x = self.output(x)
            return x

    
    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


     # Convert arrays to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Train the model
    model.train()
    for epoch in range(10000):  # Number of epochs
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
    
    

    data = [19,22,7,8,6,8,4,5,7,6,4,4,7,7,8,2,1,1,2,0,3,14,14,19,7,14,9,7,2,4,5,10,4,0,1,4,8,4,5,2,6,1,2,2,1,3]
    
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