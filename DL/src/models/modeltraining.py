import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle

# -----------------------------------------
# Define Model Classes (SimpleRNN, SimpleLSTM)
# -----------------------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------------------
# Train Model Function
# -----------------------------------------
def train_model(data_path, model_path, scaler_path, model_type='rnn', coin=None):
    data_path = Path(data_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

   
    if coin is None:
        coin_symbol = data_path.stem.split('_')[0].upper()
    else:
        coin_symbol = coin

    df = pd.read_csv(data_path)

    # -----------------------------------------
    # Create Sequences Function
    # -----------------------------------------
    def create_sequences(X_data, y_data, seq_length):
        xs, ys = [], []
        for i in range(len(X_data) - seq_length):
            xs.append(X_data[i:i+seq_length])
            ys.append(y_data[i+seq_length])
        return np.array(xs), np.array(ys)

    # -----------------------------------------
    # Data Preparation
    # -----------------------------------------
    x_cols = ['open', 'high', 'low', 'volume', 
              'quote_asset_volume', 'number_of_trades', 
              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    y_col = 'target'

    # -----------------------------------------
    # Scale the data
    # -----------------------------------------
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    df[x_cols] = x_scaler.fit_transform(df[x_cols])
    df[[y_col]] = y_scaler.fit_transform(df[[y_col]])

    # -----------------------------------------
    # Create sequences
    # -----------------------------------------
    sequence_length = 10
    X_data = df[x_cols].values
    y_data = df[y_col].values
    X_seq, y_seq = create_sequences(X_data, y_data, sequence_length)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    # -----------------------------------------
    # Split data into training and testing sets
    # -----------------------------------------
    train_size = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------------------
    # Model Initialization
    # -----------------------------------------
    input_size = len(x_cols)
    hidden_size = 64
    output_size = 1
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'rnn':
        model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)
    elif model_type == 'lstm':
        model = SimpleLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    else:
        raise ValueError("Invalid model type. Choose either 'rnn' or 'lstm'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------
    # Training Loop
    # -----------------------------------------
    best_test_loss = float('inf')
    best_model_state = None

    print(f"Training {model_type.upper()} model for {coin_symbol} on {device}")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # -----------------------------------------
        # Testing phase
        # -----------------------------------------
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()

    # -----------------------------------------
    # Final Evaluation
    # -----------------------------------------
    model.load_state_dict(best_model_state)
    model.eval()
    
    # -----------------------------------------
    # Save Model and Scalers
    # -----------------------------------------
    if model_path.suffix != '.pth':
        model_path = model_path.with_suffix('.pth')
    torch.save(model, model_path)
    
    if scaler_path.suffix != '.pkl':
        scaler_path = scaler_path.with_suffix('.pkl')
    
    scaler_dict = {'x_scaler': x_scaler, 'y_scaler': y_scaler}
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
