import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta

def get_sp500_companies():
    """Scrapes S&P 500 company tickers from Wikipedia."""
    print("Fetching S&P 500 company list...")
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(sp500_url, headers=headers)
    sp500_df = pd.read_html(response.text, header=0)[0]
    tickers = sp500_df['Symbol'].tolist()
    # Replace dots with hyphens for yfinance compatibility
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Loaded {len(tickers)} tickers.")
    return tickers

def download_historical_prices(tickers, period='1y', interval='1d'):
    """Downloads historical close prices for given tickers."""
    print("Downloading historical prices... This may take a while.")
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Approx 1 year
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval=interval)
            if not hist.empty:
                data[ticker] = hist['Close']
                print(f"Downloaded data for {ticker}")
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    # Create a DataFrame with dates as index and tickers as columns
    df = pd.DataFrame(data)
    df.index.name = 'Date'
    df = df.dropna(axis=1, how='all')  # Drop tickers with no data
    return df

def save_prices_data(df, path='data/historical_prices.parquet'):
    """Saves the prices DataFrame to a Parquet file."""
    if df.empty:
        print("No price data downloaded. Nothing to save.")
        return
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_parquet(path)
    print(f"Price data saved to {path} with shape {df.shape}")

if __name__ == "__main__":
    tickers = get_sp500_companies()
    if tickers:
        prices_df = download_historical_prices(tickers)
        save_prices_data(prices_df)
