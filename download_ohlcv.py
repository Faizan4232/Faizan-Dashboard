# download_ohlcv.py (FIXED)
import yfinance as yf
import pandas as pd
import os

def download_ohlcv(ticker="^GSPC", period="5y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,   # 🔑 FIX
        progress=True
    )

    df.reset_index(inplace=True)

    # Handle MultiIndex columns safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/historical_ohlcv.parquet", index=False)

    print("✅ Saved data/historical_ohlcv.parquet")
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    download_ohlcv()
