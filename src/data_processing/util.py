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



