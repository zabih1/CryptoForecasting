# ================================================================
# 📌 Import Required Libraries
# ================================================================
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

# ================================================================
# 📌 Define Model Classes: RNN & LSTM
# ================================================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ================================================================
# 📌 Train Model Function
# ================================================================
def train_model(data_path, model_path, scaler_path, model_type='rnn', coin=None):
    data_path, model_path, scaler_path = map(Path, [data_path, model_path, scaler_path])

    df = pd.read_csv(data_path)

    x_cols = ["open", "high", "low", "volume", "quote_asset_volume",
              "number_of_trades", "taker_buy_base_asset_volume", "average_price", "price_change"]
    y_col = 'target_close'

    sequence_length = 30
    train_split_index = int(0.8 * len(df))

    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_scaler.fit(df.iloc[:train_split_index][x_cols])
    y_scaler.fit(df.iloc[:train_split_index][[y_col]])

    df[x_cols] = x_scaler.transform(df[x_cols])
    df[[y_col]] = y_scaler.transform(df[[y_col]])

    # 🔄 Create Sequences
    def create_sequences(X, y, seq_length):
        return np.array([X[i:i + seq_length] for i in range(len(X) - seq_length)]), \
               np.array([y[i + seq_length] for i in range(len(y) - seq_length)])

    X_data, y_data = df[x_cols].values, df[y_col].values
    X_seq, y_seq = create_sequences(X_data, y_data, sequence_length)

    train_size = train_split_index - sequence_length
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_test, y_test = X_seq[train_size:], y_seq[train_size:]

    X_train, y_train = map(torch.tensor, (X_train, y_train))
    X_test, y_test = map(torch.tensor, (X_test, y_test))

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train.float(), y_train.float()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test.float(), y_test.float()), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = SimpleRNN if model_type == 'rnn' else SimpleLSTM
    model = model_cls(len(x_cols), 128, 1, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003 if model_type == 'rnn' else 0.001)
    criterion = nn.MSELoss()

    # =======================================
    #          🚀 Training Loop
    # =======================================

    best_model_state, best_loss = None, float('inf')

    for epoch in range(10 if model_type == 'lstm' else 130):
        model.train()
        train_loss = sum(criterion(model(batch_X.to(device)).squeeze(), batch_y.to(device)).item()
                         for batch_X, batch_y in train_loader) / len(train_loader)

        model.eval()
        test_loss = sum(criterion(model(batch_X.to(device)).squeeze(), batch_y.to(device)).item()
                        for batch_X, batch_y in test_loader) / len(test_loader)

        if test_loss < best_loss:
            best_loss, best_model_state = test_loss, model.state_dict()

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path.with_suffix('.pth'))
    with open(scaler_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    # =======================================
    #           📊 Evaluate Model
    # =======================================
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds.append(model(batch_X.to(device)).squeeze().cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    preds, targets = np.concatenate(preds), np.concatenate(targets)
    preds, targets = y_scaler.inverse_transform(preds.reshape(-1, 1)), y_scaler.inverse_transform(targets.reshape(-1, 1))

    print(f"\nRMSE: {np.sqrt(mean_squared_error(targets, preds)):.4f}")
    print(f"MAE: {mean_absolute_error(targets, preds):.4f}")
