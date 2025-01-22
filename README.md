# CryptoForecasting

CryptoForecasting is a machine learning project aimed at predicting cryptocurrency prices for Bitcoin (BTC) and Ethereum (ETH). The project leverages advanced data processing techniques and machine learning models to provide accurate forecasts based on historical market data.

---

## Project Structure

```

CryptoForecasting/
├── artifacts/
├── data/
│   ├── raw_data/
│   ├── processed_data/
├── notebook/
├── src/
│   ├── data_processing/
│   │   ├── util.py
│   │   ├── preprocessing.py
│   ├── models/
│   │   ├── modeltraining.py
│   │   ├── testing.py
│   ├── prediction_pipeline/
│       ├── btc_prediction.py
│       ├── eth_prediction.py
│       ├── prediction_pipeline.py
├── .gitignore
├── README.md
├── requirements.txt
├── inference.py

```

---

## Features

- **BTC and ETH Price Prediction**: Predicts future prices for Bitcoin and Ethereum.
- **Data Processing**: Includes utility scripts for preprocessing and managing raw and processed data.
- **Model Training and Evaluation**: Train and test machine learning models using `modeltraining.py` and `testing.py`.
- **Prediction Pipelines**: Ready-to-use pipelines for generating cryptocurrency price forecasts.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/zabih1/CryptoForecasting.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CryptoForecasting
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model
To train the model, modify and run the `modeltraining.py` script in the `src/models` folder.

### Predict Cryptocurrency Prices
Use the prediction pipelines:

- **Coin Prediction**:
  ```bash
  python src/prediction_pipeline/prediction_pipeline.py
  ```

### Running Inference
Use the `inference.py` script to generate `close price ` predictions from saved models:
```bash
python inference.py
```

---

## Folder Descriptions

### Artifacts
- Stores trained models for future inference and evaluation.

### Data
- **raw_data**: Contains raw market data.
- **processed_data**: Contains preprocessed data ready for model training.

### Notebook
- Jupyter notebooks for exploratory data analysis (EDA) and experimentation.

### Source Code
- **data_processing**: Contains scripts for data preprocessing and utilities.
- **models**: Scripts for training and testing machine learning models.
- **prediction_pipeline**: Scripts for generating cryptocurrency predictions.

---

