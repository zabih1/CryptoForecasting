# from src.data_processing.util import get_data, save_to_csv
# from src.models.modeltraining import train_model



import sys
import os

# Add paths for custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from src.data_processing.util import get_data, save_to_csv
from src.data_processing.preprocessing import process_data, save_processed_data
from src.models.modeltraining import train_model


def btc_prediction_pipeline():
    # Define paths and parameters
    raw_data_base_path = 'data/raw_data'
    processed_data_base_path = 'data/processed_data'
    symbol = 'BTCUSDT'
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
    model_path = 'artifacts/btc_model.pkl'
    trained_model = train_model(processed_data_path, model_path, model_type='xgboost')
    print(f"Bitcoin model trained and saved at: {model_path}")



# # Run the pipeline
# if __name__ == "__main__":
#     btc_prediction_pipeline()
