from src.data_processing.util import get_data, save_to_csv
from src.data_processing.preprocessing import process_data, save_processed_data
from src.models.modeltraining import train_model
import os

def eth_prediction_pipeline():
    # Define paths and parameters
    raw_data_base_path = 'data/raw_data'
    processed_data_base_path = 'data/processed_data'
    symbol = 'ETHUSDT'
    interval = '1d'
    
    # Step 1: Fetch raw data
    data = get_data(symbol, interval)
    raw_data_path = save_to_csv(data, symbol, interval, raw_data_base_path)
    print(f"Raw data saved at: {raw_data_path}")
    
    # Step 2: Process the data
    processed_data = process_data(data)
    processed_data_path = save_processed_data(processed_data, processed_data_base_path, symbol, interval)
    print(f"Processed data saved at: {processed_data_path}")
    
    # Step 3: Train the model
    model_path = 'artifacts/eth_model.pkl'
    trained_model = train_model(processed_data_path, model_path, model_type='xgboost')
    print(f"Ethereum model trained and saved at: {model_path}")
