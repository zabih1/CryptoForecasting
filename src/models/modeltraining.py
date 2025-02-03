import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def train_model(data_path, model_path, model_type='linear'):

    data_path = Path(data_path)
    model_path = Path(model_path)
    
    data = pd.read_csv(data_path)

    features = ['Open', 'High', 'Low', 'Volume', 
                'Average Price', 'Price Change']
    
    X = data[features]
    y = data['Close']

    split_ratio = 0.8  # 80% train, 20% test
    split_point = int(len(data) * split_ratio)

    train = data.iloc[:split_point]  
    test = data.iloc[split_point:]  

    train_X = train[features]
    train_y = train['Close']

    test_X = test[features]
    test_y = test['Close']

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)  

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
    return model
