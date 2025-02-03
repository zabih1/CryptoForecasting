import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_model(data_path, model_path, model_type='linear'):
    data_path = Path(data_path)
    model_path = Path(model_path)

    data = pd.read_csv(data_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
       'Number of Trades', 'Taker Buy Base Volume', 'Taker Buy Quote Volume',
       'Average Price', 'Price Change', 'year', 'month', 'day']
    
    X = data[features]
    y = data['Close']

    # Split data (time series method)
    split_ratio = 0.8  # 80% train, 20% test
    split_point = int(len(data) * split_ratio)

    train_X = X[:split_point]
    train_y = y[:split_point]
    test_X = X[split_point:]
    test_y = y[split_point:]

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError('Unsupported model type. Choose "linear" or "xgboost".')

    model.fit(train_X, train_y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model trained and saved at: {model_path}")

    evaluation(test_X, test_y, model_path)


def evaluation(test_X, test_y, model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(test_X)

    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print(f"Model: {model_path.name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    print("-" * 50)

    return mse, mae, r2


