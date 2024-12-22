import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import psycopg2

# Function to connect to the database and fetch data
def get_data_for_training():
    conn = psycopg2.connect(
        host="localhost",
        dbname="mnrj",
        user="postgres",
        password="gwdb",
        port=5600
    )
    df = pd.read_sql("SELECT * FROM game_stats", conn)
    conn.close()
    return df

# Function to parse data and prepare for LSTM input
def parseData(df, player_name):
    label_encoder = LabelEncoder()
    df['opponent_encoded'] = label_encoder.fit_transform(df['opponent'])

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

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# LSTM Network Definition
class GameStatsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GameStatsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

# Main execution function
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Check if MPS is available
    print(f"Using {device} for computation.")

    df = get_data_for_training()
    X, y = parseData(df, "Jayson Tatum")  # Replace with your player's name

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate the correct input dimension
    input_dim = X_scaled.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, device=device).unsqueeze(1)  # Moving data to MPS device
    X_test = torch.tensor(X_test, device=device).unsqueeze(1)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    model = GameStatsLSTM(input_dim, 100, 1, 1).to(device)  # Move model to MPS device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 50000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()  # Squeeze the output to remove extra dimensions
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test MSE: {test_loss.item()}')
    
    new_data = [19,22,7,8,6,8,4,5,7,6,4,4,7,7,8,2,1,1,2,0,3,14,14,19,7,14,9,7,2,4,5,10,4,0,1,4,8,4,5,2,6,1,2,2,1,3]
    new_data = np.array(new_data).reshape(1, -1)  # Reshape data if necessary (depends on model input requirements)

    # Scale the data as per the training data scaling
    new_data_scaled = scaler.transform(new_data)

    # Convert scaled data to tensor and add necessary dimensions if required by the model
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # Add .unsqueeze(1) if needed

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        prediction = model(new_data_tensor)
        print("Predicted output:", prediction.squeeze().item())  # Adjust based on your output format


if __name__ == "__main__":
    main()
