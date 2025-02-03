import os
import pandas as pd

def load_raw_data(base_path, symbol, interval):
    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)

def process_data(df):
    df = df.copy()
    df['Open'] = pd.to_numeric(df['Open'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Volume'] = pd.to_numeric(df['Volume'])
    df['Average Price'] = (df['High'] + df['Low']) / 2
    df['Price Change'] = df['Close'] - df['Open']
    df['year'] = df['Open Time'].dt.year  
    df['month'] = df['Open Time'].dt.month  
    df['day'] = df['Open Time'].dt.day
    return df

def save_processed_data(df, base_path, symbol, interval):
    filename = f"{symbol.lower()}_{interval}_processed.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path



