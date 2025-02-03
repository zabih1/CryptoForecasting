import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

def evaluate_model(model_path, data_path):
    model_path = Path(model_path)
    data_path = Path(data_path)

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    data = pd.read_csv(data_path)

    features = ['Open', 'High', 'Low', 'Volume', 'Average Price', 'Price Change']
    target = 'Close'  

    X = data[features]
    y = data[target]

    split_ratio = 0.8  # 80% train, 20% test (time-based split)
    split_point = int(len(data) * split_ratio)

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)  # Apply scaler to test features only

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_path.name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    print("-" * 50)

if __name__ == "__main__":
    models = [
        "artifacts/btcusdt_1d_linear_model.pkl",
        "artifacts/btcusdt_1d_xgboost_model.pkl",
        "artifacts/ethusdt_1d_linear_model.pkl",
        "artifacts/ethusdt_1d_xgboost_model.pkl",
    ]

    test_data_path = "data/processed_data/btcusdt_1d_processed.csv"  

    for model_path in models:
        evaluate_model(model_path, test_data_path)
