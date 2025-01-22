import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from src.data_processing.util import get_data, save_to_csv
from src.data_processing.preprocessing import process_data, save_processed_data
from src.models.modeltraining import train_model


def btc_prediction_pipeline(symbol, interval):
    """
    Full prediction pipeline for a given symbol and interval.
    """
    base_dir = Path('.')
    raw_data_base_path = base_dir / 'data/raw_data'
    processed_data_base_path = base_dir / 'data/processed_data'
    artifacts_dir = base_dir / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = get_data(symbol, interval)
    raw_data_path = save_to_csv(data, symbol, interval, raw_data_base_path)

    processed_data = process_data(data)
    processed_data_path = save_processed_data(processed_data, processed_data_base_path, symbol, interval)

    models = ["linear", "xgboost"]
    for model_type in models:
        model_file_name = f"{symbol.lower()}_{interval}_{model_type}_model.pkl"
        model_path = artifacts_dir / model_file_name
        train_model(processed_data_path, model_path, model_type=model_type)
        print("="*50)


if __name__ == "__main__":
    btc_prediction_pipeline(symbol='BTCUSDT', interval='1d')
