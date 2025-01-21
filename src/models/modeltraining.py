import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
from pathlib import Path

def train_model(data_path, model_path, model_type='linear'):

    data_path = Path(data_path)
    model_path = Path(model_path)
    
    data = pd.read_csv(data_path)
    
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError('Unsupported model type. Choose "linear" or "xgboost".')

    model.fit(X_train, y_train)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model trained and saved at: {model_path}")
    return model



# BASE_DIR = Path("crypto_forecasting_project")
# PROCESSED_DATA_DIR = BASE_DIR / "data/processed_data"
# ARTIFACTS_DIR = BASE_DIR / "artifacts"

# btc_data_path = "data/processed_data/btcusdt_1d_processed.csv"
# btc_model_path = "artifacts/btc_model.pkl"

# train_model(btc_data_path, btc_model_path, model_type='xgboost')
