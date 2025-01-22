import pandas as pd
import requests
import os

def get_data(symbol, interval, limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
               'Close Time', 'Quote Asset Volume', 'Number of Trades', 
               'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore']
    df = pd.DataFrame(data, columns=columns)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    return df

def save_to_csv(df, symbol, interval, base_path):
    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

# base_path = os.path.join("data", "raw_data")

# btc_data = get_data('BTCUSDT', '1d')
# eth_data = get_data('ETHUSDT', '1d')

# btc_path = save_to_csv(btc_data, 'BTCUSDT', '1d', base_path)
# eth_path = save_to_csv(eth_data, 'ETHUSDT', '1d', base_path)

# print(f"Bitcoin data saved at: {btc_path}")
# print(f"Ethereum data saved at: {eth_path}")
