import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    """
    Load a model from the specified file path using pickle.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        return model

def predict_close_price(model_path, input_data):
    """
    Predict the closing price using the specified model and input data.
    """
    model = load_model(model_path)
    df = pd.DataFrame([input_data])  
    prediction = model.predict(df)[0]  
    return prediction

if __name__ == "__main__":
    btc_linear_model_path = "artifacts/btcusdt_1d_linear_model.pkl"
    eth_linear_model_path = "artifacts/ethusdt_1d_linear_model.pkl"

    
    btc_input_data = {
        "Open": 42000.0,          # Opening price of the asset
        "High": 43500.0,          # Highest price during the trading period
        "Low": 41000.0,           # Lowest price during the trading period
        "Volume": 25000.0,        # Volume of the asset traded
        "Average Price": 42500.0, # Average price of the asset during the period
        "Price Change": -500.0    # Difference between Close and Open prices
    }
    
    eth_input_data = {
        "Open": 3200.0,          
        "High": 3350.0,          
        "Low": 3100.0,           
        "Volume": 18000.0,       
        "Average Price": 3250.0, 
        "Price Change": -50.0    
    }

    
    btc_linear_prediction = predict_close_price(btc_linear_model_path, btc_input_data)
    print(f"Predicted Bitcoin Closing Price (Linear Model): {btc_linear_prediction}")
    
    eth_linear_prediction = predict_close_price(eth_linear_model_path, eth_input_data)
    print(f"Predicted Ethereum Closing Price (Linear Model): {eth_linear_prediction}")
