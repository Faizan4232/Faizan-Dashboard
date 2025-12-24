"""
download_data.py - Download S&P 500 historical OHLCV data from Yahoo Finance

Usage:

python download_data.py [--period 1y] [--interval 1d] [--output data/historical_ohlcv.parquet]

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

    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    response = requests.get(sp500_url, headers=headers)
    response.raise_for_status()

    sp500_df = pd.read_html(response.text, header=0)[0]
    tickers = sp500_df["Symbol"].tolist()
    # Yahoo Finance uses '-' instead of '.' in some tickers (e.g., BRK.B -> BRK-B)
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    print(f"Loaded {len(tickers)} tickers.")
    return tickers


def download_historical_ohlcv(tickers, period="1y", interval="1d", delay=0.2):
    """Downloads historical OHLCV data for given tickers."""
    print(f"Downloading historical OHLCV (period={period}, interval={interval})... This may take a while.")
    data = {}
    errors = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) if period == "1y" else None

    needed_cols = ["Open", "High", "Low", "Close", "Volume"]

    for ticker in tqdm(tickers, desc="Tickers Downloaded"):
        try:
            stock = yf.Ticker(ticker)
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                hist = stock.history(period=period, interval=interval)

            # Keep only OHLCV
            if not hist.empty and all(col in hist.columns for col in needed_cols):
                hist = hist[needed_cols]
                data[ticker] = hist
            else:
                errors.append(ticker)
        except Exception:
            errors.append(ticker)
        time.sleep(delay)

    if not data:
        print("No data downloaded.")
        return pd.DataFrame()

    # Combine into one DataFrame with MultiIndex columns (ticker, field)
    combined = pd.concat(data, axis=1)  # outer keys=ticker
    combined.index.name = "Date"

    # Drop tickers that ended up completely NaN
    combined = combined.dropna(axis=1, how="all")

    # Count unique tickers from MultiIndex columns
    unique_tickers = sorted({c[0] for c in combined.columns})
    print(f"\nSuccessfully downloaded: {len(unique_tickers)} tickers. Missing or failed: {len(errors)}.")
    if errors:
        print("Tickers not found or errored:", ", ".join(errors))

    return combined


def save_ohlcv_data(df, path="data/historical_ohlcv.parquet"):
    """Saves the OHLCV DataFrame to a Parquet file."""
    if df.empty:
        print("No OHLCV data downloaded. Nothing to save.")
        return

    outdir = os.path.dirname(path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    df.to_parquet(path)
    print(f"OHLCV data saved to {path} with shape {df.shape}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--period",
        default="1y",
        help="History period for download: e.g. '1y', '6mo', '5d', etc.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval: '1d', '1h', etc.",
    )
    parser.add_argument(
        "--output",
        default="data/historical_ohlcv.parquet",
        help="Output Parquet file path",
    )

    args = parser.parse_args()

    tickers = get_sp500_companies()
    if tickers:
        ohlcv_df = download_historical_ohlcv(
            tickers,
            period=args.period,
            interval=args.interval,
            delay=0.2,
        )
        save_ohlcv_data(ohlcv_df, path=args.output)
