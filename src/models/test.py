import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

def evaluate_model(model_path, test_data_path):
    model_path = Path(model_path)
    test_data_path = Path(test_data_path)
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['Open', 'High', 'Low', 'Volume']]
    y_test = test_data['Close']
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"MSE": mse, "MAE": mae, "R2": r2}

if __name__ == "__main__":

    btc_model_path = "artifacts/eth_model.pkl"  
    btc_test_data_path = "data/processed_data/btcusdt_1d_processed.csv"  

    metrics = evaluate_model(btc_model_path, btc_test_data_path)
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
