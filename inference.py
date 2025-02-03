import pandas as pd
import pickle
import warnings
import requests
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(model_path):
    """Load a model from the specified file path using pickle."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def get_binance_data(symbol, interval, limit=1):
    """Fetch the latest market data from Binance API."""
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    try:
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()[0]

        return {
            "Open Time": int(data[0]),  
            "Open": float(data[1]),
            "High": float(data[2]),
            "Low": float(data[3]),
            "Close": float(data[4]),
            "Volume": float(data[5]),
            "Quote Asset Volume": float(data[7]),
            "Number of Trades": int(data[8]),
            "Taker Buy Base Volume": float(data[9]),
            "Taker Buy Quote Volume": float(data[10]),
            "Average Price": (float(data[1]) + float(data[4])) / 2,  # (Open + Close) / 2
            "Price Change": float(data[4]) - float(data[1])  # Close - Open
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise




def process_input_data(input_data):
    """Process the input data to match the model's expected feature format."""
    df = pd.DataFrame([input_data])

    df = df.drop(columns=["Open Time"], errors="ignore")

    df['Average Price'] = (df['High'] + df['Low']) / 2
    df['Price Change'] = df['Close'] - df['Open']

    timestamp = pd.to_datetime(input_data["Open Time"], unit='ms')
    df['year'] = timestamp.year
    df['month'] = timestamp.month
    df['day'] = timestamp.day

    return df



def predict_close_price(model, input_data):
    """Predict the closing price using the specified model and processed input data."""
    df = process_input_data(input_data)
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    btc_linear_model_path = "artifacts/btcusdt_1d_linear_model.pkl"
    eth_linear_model_path = "artifacts/ethusdt_1d_linear_model.pkl"

    try:
        btc_input_data = get_binance_data("BTCUSDT", "1d")
        eth_input_data = get_binance_data("ETHUSDT", "1d")

        btc_linear_model = load_model(btc_linear_model_path)
        eth_linear_model = load_model(eth_linear_model_path)

        btc_linear_prediction = predict_close_price(btc_linear_model, btc_input_data)
        eth_linear_prediction = predict_close_price(eth_linear_model, eth_input_data)

        print(f"Predicted Bitcoin Closing Price (Linear Model): {btc_linear_prediction}")
        print(f"Predicted Ethereum Closing Price (Linear Model): {eth_linear_prediction}")

        logging.info(f"BTCUSDT Prediction: {btc_linear_prediction}")
        logging.info(f"ETHUSDT Prediction: {eth_linear_prediction}")

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print(f"Error: {e}")
