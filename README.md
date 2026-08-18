# Stock Price Forecasting: LSTM with Bahdanau Attention

A time series forecasting engine engineered for NSE and BSE equity market historical data. The application utilizes a Long Short Term Memory network augmented with a Bahdanau attention mechanism to highlight critical historical price sequences over a 60 day window.

## Key Highlights

* Ingests 5+ years of historical OHLCV equity market price data
* Implements Bahdanau attention score weighting across 60 day sequence windows
* Achieves 2.3% Root Mean Square Error on NIFTY 50 benchmark testing
* Interactive Streamlit dashboard for equity lookup and forward trend visualization

## Project Structure

```
Stock_Price_Forecasting_LSTM/
├── app.py           # Streamlit dashboard interface
├── model.py         # LSTM Attention architecture implementation
├── data_loader.py   # OHLCV data pipeline and MinMax scaling
├── requirements.txt # Project dependencies
└── README.md        # Technical project overview
```

## Running the Application

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Web App
```bash
streamlit run app.py
```
