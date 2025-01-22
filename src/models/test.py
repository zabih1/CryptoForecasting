import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path



def evaluate_model(model_path, data_path):
    model_path = Path(model_path)

    data_path = Path(data_path)


    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    data = pd.read_csv(data_path)

    features = ['Open', 'High', 'Low', 'Volume', 
                'Average Price', 'Price Change']
    target = 'Close' 

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()


    X_test = scaler.fit_transform(X_test)
    y_test = scaler.fit_transform(y_test.values.reshape(-1,1))

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":

    btc_model_path = "artifacts/eth_model.pkl"  
    btc_test_data_path = "data/processed_data/btcusdt_1d_processed.csv"  

    metrics = evaluate_model(btc_model_path, btc_test_data_path)
