import sys
import os
from pathlib import Path

#---------------------Modify System Path-------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

#---------------------Import Required Modules-------------------------------
from src.data_processing.util import get_data, save_to_csv
from src.data_processing.preprocessing import process_data, save_processed_data
from src.models.modeltraining import train_model

#---------------------ETH Prediction Pipeline-------------------------------
def eth_prediction_pipeline(symbol, interval, start_date, end_date):
    """
    Full prediction pipeline for Ethereum (ETH) based on the given interval.
    """
    base_dir = Path('./ML')
    raw_data_base_path = base_dir / 'data/raw_data'
    processed_data_base_path = base_dir / 'data/processed_data'
    artifacts_dir = base_dir / 'artifacts'

    # Create artifacts directory if it doesn't exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for model and scaler inside artifacts directory
    model_dir = artifacts_dir / 'model'
    scaler_dir = artifacts_dir / 'scaler'

    # Create directories for models and scalers
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    #---------------------Fetch and Save Raw Data-------------------------------
    data = get_data(symbol, interval, start_date, end_date)
    raw_data_path = save_to_csv(data, symbol, interval, raw_data_base_path)

    #---------------------Process and Save Data-------------------------------
    processed_data = process_data(data)
    processed_data_path = save_processed_data(processed_data, processed_data_base_path, symbol, interval)

    #---------------------Train and Save Models-------------------------------
    models = ["linear", "xgboost"]
    for model_type in models:
        # Define the paths for model and scaler
        model_file_name = f"{symbol.lower()}_{interval}_{model_type}_model.pkl"
        model_path = model_dir / model_file_name

        scaler_file_name = f"{symbol.lower()}_{interval}_scaler.pkl"
        scaler_path = scaler_dir / scaler_file_name  

        # Train and save the model and scaler
        train_model(processed_data_path, model_path, scaler_path, model_type=model_type)
        print("=" * 50)
