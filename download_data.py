"""
download_data.py - Download S&P 500 historical close prices from Yahoo Finance

Usage:
    python download_data.py [--period 1y] [--interval 1d] [--output data/historical_prices.parquet]

Requires:
    pandas, yfinance, requests, tqdm
"""

import pandas as pd
import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

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
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    print(f"Loaded {len(tickers)} tickers.")
    return tickers

def download_historical_prices(tickers, period='1y', interval='1d', delay=0.2):
    """Downloads historical close prices for given tickers."""
    print(f"Downloading historical prices (period={period}, interval={interval})... This may take a while.")
    data = {}
    errors = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) if period=='1y' else None
    for ticker in tqdm(tickers, desc='Tickers Downloaded'):
        try:
            stock = yf.Ticker(ticker)
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                hist = stock.history(period=period, interval=interval)
            if not hist.empty and 'Close' in hist.columns:
                data[ticker] = hist['Close']
            else:
                errors.append(ticker)
        except Exception as e:
            errors.append(ticker)
        time.sleep(delay)
    df = pd.DataFrame(data)
    df.index.name = 'Date'
    df = df.dropna(axis=1, how='all')
    print(f"\nSuccessfully downloaded: {df.shape[1]} tickers. Missing or failed: {len(errors)}.")
    if errors:
        print("Tickers not found or errored:", ', '.join(errors))
    return df

def save_prices_data(df, path='data/historical_prices.parquet'):
    """Saves the prices DataFrame to a Parquet file."""
    if df.empty:
        print("No price data downloaded. Nothing to save.")
        return
    outdir = os.path.dirname(path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_parquet(path)
    print(f"Price data saved to {path} with shape {df.shape}")
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', default='1y', help="History period for download: e.g. '1y', '6mo', '5d', etc.")
    parser.add_argument('--interval', default='1d', help="Data interval: '1d', '1h', etc.")
    parser.add_argument('--output', default='data/historical_prices.parquet', help="Output Parquet file path")
    args = parser.parse_args()
    
    tickers = get_sp500_companies()
    if tickers:
        prices_df = download_historical_prices(tickers, period=args.period, interval=args.interval, delay=0.2)
        save_prices_data(prices_df, path=args.output)
