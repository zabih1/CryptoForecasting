import pandas as pd
import joblib

def load_model(model_path):
  
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")

def predict_close_price(model_path, input_data):
  
    model = load_model(model_path)
    df = pd.DataFrame([input_data])  
    prediction = model.predict(df)[0] 
    return prediction

if __name__ == "__main__":
    
    btc_model_path = "artifacts/btc_model.pkl"
    eth_model_path = "artifacts/eth_model.pkl"
    
    btc_input_data = {
        "Open": 43000.0,
        "High": 45000.0,
        "Low": 42000.0,
        "Volume": 30000.0
    }
    
    eth_input_data = {
        "Open": 3000.0,
        "High": 3100.0,
        "Low": 2900.0,
        "Volume": 15000.0
    }
    
    btc_prediction = predict_close_price(btc_model_path, btc_input_data)
    print(f"Predicted Bitcoin Closing Price: {btc_prediction}")
    
    eth_prediction = predict_close_price(eth_model_path, eth_input_data)
    print(f"Predicted Ethereum Closing Price: {eth_prediction}")
