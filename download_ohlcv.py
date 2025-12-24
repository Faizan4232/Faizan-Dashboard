# download_ohlcv.py
import yfinance as yf
import pandas as pd
import os

def download_ohlcv(ticker="^GSPC", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/historical_ohlcv.parquet")
    print("Saved data/historical_ohlcv.parquet")

if __name__ == "__main__":
    download_ohlcv()
