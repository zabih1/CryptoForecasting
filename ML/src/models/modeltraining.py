import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#-------------------------------------Train Model---------------------------------------------
def train_model(data_path, model_path, scaler_path, model_type='linear'):
    data_path = Path(data_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    df = pd.read_csv(data_path)

    features = ['open', 'high', 'low', 'volume', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'price_range', 'close_to_open']
    
    X = df[features]
    y = df['target']  

    split_ratio = 0.8  
    split_point = int(len(df) * split_ratio)

    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    #---------------------Feature Scaling and Save the scaler-------------------------------

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler_X, file)
    print(f"Scaler saved at: {scaler_path}")

    #---------------------Initialize and Train Model and Save the model---------------------

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError('Unsupported model type. Choose "linear" or "xgboost".')

    model.fit(X_train_scaled, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model trained and saved at: {model_path}")

    evaluation(X_test_scaled, y_test, model_path)


#-----------------------------Evaluation of the models------------------------------------

def evaluation(X_test_scaled, y_test, model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Model: {model_path.name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print("-" * 50)

    return mse, mae, rmse
