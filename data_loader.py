import numpy as np
import pandas as pd

def fetch_stock_data(ticker="RELIANCE.NS", period="5y"):
    """
    Simulates / ingests OHLCV financial data for NSE / BSE equities
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=1250, freq='B')
    
    # Generate realistic random walk stock price trajectory
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    price_path = 2000.0 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': price_path * (1 - np.random.uniform(0.001, 0.005, len(dates))),
        'High': price_path * (1 + np.random.uniform(0.002, 0.01, len(dates))),
        'Low': price_path * (1 - np.random.uniform(0.002, 0.01, len(dates))),
        'Close': price_path,
        'Volume': np.random.randint(500000, 5000000, len(dates))
    })
    df.set_index('Date', inplace=True)
    return df

def preprocess_sequences(df, sequence_length=60):
    prices = df['Close'].values
    min_val = np.min(prices)
    max_val = np.max(prices)
    scaled = (prices - min_val) / (max_val - min_val + 1e-8)
    return scaled, min_val, max_val
