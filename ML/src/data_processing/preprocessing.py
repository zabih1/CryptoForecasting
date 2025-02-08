import os
import pandas as pd

#---------------------Load Raw Data-------------------------------
def load_raw_data(base_path, symbol, interval):
    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)

#---------------------Process Data-------------------------------
def process_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    
    df['price_range'] = df['high'] - df['low']
    df['close_to_open'] = df['close'] - df['open']
    df['target'] = df['close'].shift(-1)

    df = df.dropna()
    
    return df

#---------------------Save Processed Data-------------------------------
def save_processed_data(df, base_path, symbol, interval):
    filename = f"{symbol.lower()}_{interval}_processed.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
