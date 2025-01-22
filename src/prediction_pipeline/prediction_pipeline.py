import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.prediction_pipeline.eth_prediction import eth_prediction_pipeline
from src.prediction_pipeline.btc_prediction import btc_prediction_pipeline

def run_prediction_pipelines():

    btc_prediction_pipeline(symbol='BTCUSDT', interval='1d')
    eth_prediction_pipeline(symbol='ETHUSDT', interval='1d')

if __name__ == "__main__":
    run_prediction_pipelines()
