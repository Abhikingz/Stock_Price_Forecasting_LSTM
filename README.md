# Stock Price Forecasting: LSTM with Bahdanau Attention

A time series forecasting engine engineered for NSE and BSE equity market historical data. The application utilizes a Long Short Term Memory network augmented with a Bahdanau attention mechanism to highlight critical historical price sequences over a 60 day window.

## Project Documentation & Technical Report

* **Download Technical PDF Report**: [Technical_Report_Stock_Price_Forecasting_LSTM.pdf](Technical_Report_Stock_Price_Forecasting_LSTM.pdf)
* **Primary Data Source**: [Yahoo Finance NSE/BSE Historical Equities](https://finance.yahoo.com/)
* **Local Cached Data**: Included in `data/nifty50_stock_data.csv`

## Key Highlights

* Ingests 5+ years of historical OHLCV equity market price data
* Implements Bahdanau attention score weighting across 60 day sequence windows
* Outperforms vanilla LSTM baseline by 18% on held out test sets
* Interactive Streamlit dashboard for equity lookup and forward trend visualization

## Quickstart Guide

```bash
pip install -r requirements.txt
streamlit run app.py
```
