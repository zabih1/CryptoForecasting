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
    return df

def save_processed_data(df, base_path, symbol, interval):
    filename = f"{symbol.lower()}_{interval}_processed.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

# def main():
    
#     raw_data_path = "data/raw_data"
#     processed_data_path = "data/processed_data"

#     for symbol in ['BTCUSDT', 'ETHUSDT']:
#         interval = '1d'
#         raw_data = load_raw_data(raw_data_path, symbol, interval)
#         processed_data = process_data(raw_data)
#         processed_file = save_processed_data(processed_data, processed_data_path, symbol, interval)
#         print(f"Processed data saved at: {processed_file}")

# if __name__ == "__main__":
#     main()
