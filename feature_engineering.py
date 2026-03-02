# ==================================================
# ADVANCED FEATURE ENGINEERING
# ==================================================

import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# -------------------------------
# LOAD & CLEAN PRICE DATA
# -------------------------------
df = pd.read_parquet("data/historical_prices_fixed.parquet")
df.columns = df.columns.str.strip()

if "Price" in df.columns:
    df = df.rename(columns={"Price": "Date"})

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Date", "Close"])
df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

# ==================================================
# TECHNICAL INDICATORS
# ==================================================
def add_technical_features(df):
    g = df.groupby("Company")

    # ── Moving Averages ──────────────────────────
    df["MA_10"] = g["Close"].transform(lambda x: x.rolling(10).mean())
    df["MA_20"] = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"] = g["Close"].transform(lambda x: x.rolling(50).mean())
    df["EMA_12"] = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["EMA_26"] = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    # ── MACD ─────────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df.groupby("Company")["MACD"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── RSI (14) ──────────────────────────────────
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))

    df["RSI"] = g["Close"].transform(calc_rsi)

    # ── Bollinger Bands ───────────────────────────
    df["BB_Mid"]   = df["MA_20"]
    df["BB_Std"]   = g["Close"].transform(lambda x: x.rolling(20).std())
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)
    df["BB_Pos"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ── Returns & Momentum ────────────────────────
    for period in [1, 3, 5, 10]:
        df[f"Return_{period}"]   = g["Close"].transform(lambda x: x.pct_change(period))
        df[f"Momentum_{period}"] = g["Close"].transform(lambda x: x.diff(period))

    # ── Volatility ────────────────────────────────
    df["Vol_10"] = g["Return_1"].transform(lambda x: x.rolling(10).std())
    df["Vol_20"] = g["Return_1"].transform(lambda x: x.rolling(20).std())

    # ── ATR (Average True Range) ──────────────────
    if all(c in df.columns for c in ["High", "Low"]):
        df["TR"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df.groupby("Company")["Close"].shift(1)),
                abs(df["Low"]  - df.groupby("Company")["Close"].shift(1))
            )
        )
        df["ATR_14"] = g["TR"].transform(lambda x: x.rolling(14).mean())
        df.drop(columns=["TR"], inplace=True)

    # ── Price Ratios ──────────────────────────────
    df["Price_vs_MA20"] = (df["Close"] - df["MA_20"]) / (df["MA_20"] + 1e-9)
    df["Price_vs_MA50"] = (df["Close"] - df["MA_50"]) / (df["MA_50"] + 1e-9)
    df["MA_spread"]     = (df["MA_10"] - df["MA_20"]) / (df["MA_20"] + 1e-9)

    # ── Volume Features ───────────────────────────
    if "Volume" in df.columns:
        df["Volume_MA10"]   = g["Volume"].transform(lambda x: x.rolling(10).mean())
        df["Volume_Ratio"]  = df["Volume"] / (df["Volume_MA10"] + 1e-9)
        df["Volume_Change"] = g["Volume"].transform(lambda x: x.pct_change(1))

    # ── Stochastic Oscillator ─────────────────────
    if all(c in df.columns for c in ["High", "Low"]):
        low14  = g["Low"].transform(lambda x: x.rolling(14).min())
        high14 = g["High"].transform(lambda x: x.rolling(14).max())
        df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
        df["Stoch_D"] = g["Stoch_K"].transform(lambda x: x.rolling(3).mean())

    # ── Candle Features ───────────────────────────
    if all(c in df.columns for c in ["Open", "High", "Low"]):
        df["Candle_Body"]  = abs(df["Close"] - df["Open"])
        df["Candle_Range"] = df["High"] - df["Low"]
        df["Candle_Dir"]   = np.sign(df["Close"] - df["Open"])   # +1 up, -1 down

    return df


df = add_technical_features(df)

# ==================================================
# LOAD & MERGE SENTIMENT
# ==================================================
sent_df = pd.read_parquet("data/news_sentiment.parquet")
sent_df["date"] = pd.to_datetime(sent_df["date"])
sent_df = sent_df.rename(columns={"date": "Date", "sentiment": "Sentiment"})

df = df.merge(
    sent_df[["Date", "ticker", "Sentiment"]],
    left_on=["Date", "Company"],
    right_on=["Date", "ticker"],
    how="left"
)
df["Sentiment"] = df["Sentiment"].fillna(0.0)
df.drop(columns=["ticker"], errors="ignore", inplace=True)

# ==================================================
# ADVANCED SENTIMENT FEATURES
# ==================================================
def add_sentiment_features(df):
    g = df.groupby("Company")

    # Lags
    df["Sentiment_Lag1"] = g["Sentiment"].shift(1)
    df["Sentiment_Lag2"] = g["Sentiment"].shift(2)
    df["Sentiment_Lag3"] = g["Sentiment"].shift(3)

    # Rolling stats
    df["Sentiment_MA3"]  = g["Sentiment"].transform(lambda x: x.rolling(3).mean())
    df["Sentiment_MA7"]  = g["Sentiment"].transform(lambda x: x.rolling(7).mean())
    df["Sentiment_Std5"] = g["Sentiment"].transform(lambda x: x.rolling(5).std())

    # Momentum / change
    df["Sentiment_Change"]  = g["Sentiment"].diff()
    df["Sentiment_Accel"]   = g["Sentiment_Change"].diff()   # 2nd derivative

    # Cumulative sentiment (short window)
    df["Sentiment_Cum5"]  = g["Sentiment"].transform(lambda x: x.rolling(5).sum())

    # Interaction with price momentum
    df["Sent_x_Return1"]  = df["Sentiment"] * df["Return_1"]
    df["Sent_x_Momentum"] = df["Sentiment"] * df["Momentum_5"]

    # Weighted by volume
    if "Volume" in df.columns:
        df["Sentiment_Weighted"]    = df["Sentiment"] * np.log1p(df["Volume"])
        df["Sentiment_Weighted_MA"] = g["Sentiment_Weighted"].transform(
            lambda x: x.rolling(3).mean()
        )

    # Positive / Negative sentiment indicator
    df["Sentiment_Pos"] = (df["Sentiment"] > 0).astype(int)
    df["Sentiment_Neg"] = (df["Sentiment"] < 0).astype(int)

    return df


df = add_sentiment_features(df)

# ==================================================
# CLEAN & SAVE
# ==================================================
df = df.dropna()
df = df.reset_index(drop=True)

df.to_parquet("data/master_dataset.parquet", index=False)

print("✅ Advanced Feature Engineering Completed")
print(f"Final shape : {df.shape}")
print(f"Features    : {list(df.columns)}")