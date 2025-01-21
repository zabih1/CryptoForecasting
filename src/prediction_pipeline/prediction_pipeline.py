import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.prediction_pipeline.eth_prediction import eth_prediction_pipeline
from src.prediction_pipeline.btc_prediction import btc_prediction_pipeline

def run_prediction_pipelines():
    print("Running Bitcoin prediction pipeline...")
    btc_prediction_pipeline()
    print("\nRunning Ethereum prediction pipeline...")
    eth_prediction_pipeline()

# Run both pipelines
if __name__ == "__main__":
    run_prediction_pipelines()
