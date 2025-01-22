


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_model(data_path, model_path, model_type='linear'):

    data_path = Path(data_path)
    model_path = Path(model_path)
    
    data = pd.read_csv(data_path)

    features = ['Open', 'High', 'Low', 'Volume', 
                'Average Price', 'Price Change']
    
    X = data[features]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError('Unsupported model type. Choose "linear" or "xgboost".')

    model.fit(X_train, y_train)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


    print(f"Model trained and saved at: {model_path}")
    return model




