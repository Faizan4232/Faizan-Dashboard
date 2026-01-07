# # feature_engineering.py
# """
# Feature engineering:
# - Load merged master dataset
# - Compute technical indicators
# - Create trend label (up/down)
# - Select best indicators
# - Save features and feature importance

# Input:
#     data/master_dataset.parquet

# Outputs:
#     data/features_selected.parquet
#     data/feature_importance.csv
# """

# import os
# import pandas as pd
# import ta
# from sklearn.feature_selection import SelectKBest, f_classif

# INPUT_PATH = "data/master_dataset.parquet"
# OUTPUT_FEATURES = "data/features_selected.parquet"
# OUTPUT_IMPORTANCE = "data/feature_importance.csv"

# if not os.path.exists(INPUT_PATH):
#     raise FileNotFoundError(f"{INPUT_PATH} not found. Run merge_data.py first.")

# print("📥 Loading master dataset with pandas...")
# df = pd.read_parquet(INPUT_PATH)
# print("Available columns:", list(df.columns))

# # Standardize column names
# cols = {c.lower(): c for c in df.columns}
# date_col = cols.get("date") or cols.get("timestamp") or cols.get("datetime")
# ticker_col = cols.get("ticker")
# close_col = cols.get("close") or cols.get("adjclose") or cols.get("adj_close")

# if date_col is None or ticker_col is None or close_col is None:
#     raise ValueError(
#         "Master dataset must contain date/ticker/close information. "
#         f"Found columns: {list(df.columns)}"
#     )

# df = df.rename(
#     columns={
#         date_col: "date",
#         ticker_col: "ticker",
#         close_col: "close",
#     }
# )

# if "daily_sentiment" in df.columns and "sentiment" not in df.columns:
#     df = df.rename(columns={"daily_sentiment": "sentiment"})

# df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
# df["date"] = pd.to_datetime(df["date"])

# print("🔧 Computing technical indicators...")

# def add_indicators(group):
#     c = group["close"]
#     group["rsi"] = ta.momentum.RSIIndicator(c).rsi()
#     group["ema"] = ta.trend.EMAIndicator(c).ema_indicator()
#     macd_obj = ta.trend.MACD(c)
#     group["macd"] = macd_obj.macd()
#     bb_obj = ta.volatility.BollingerBands(c)
#     group["bb"] = bb_obj.bollinger_mavg()
#     return group

# df = df.groupby("ticker", group_keys=False).apply(add_indicators)

# print("📈 Creating trend label...")
# df["next_close"] = df.groupby("ticker")["close"].shift(-1)
# df["trend"] = (df["next_close"] > df["close"]).astype(int)

# if "sentiment" in df.columns:
#     df["sentiment"] = df["sentiment"].fillna(0.0)
# else:
#     df["sentiment"] = 0.0

# df = df.dropna(subset=["rsi", "ema", "macd", "bb", "trend"])

# print("🧠 Selecting best indicators...")
# features = ["rsi", "ema", "macd", "bb"]
# X = df[features]
# y = df["trend"]

# selector = SelectKBest(score_func=f_classif, k=3)
# X_selected = selector.fit_transform(X, y)
# selected_features = X.columns[selector.get_support()].tolist()

# importance_df = pd.DataFrame(
#     {
#         "feature": features,
#         "score": selector.scores_,
#         "selected": selector.get_support(),
#     }
# ).sort_values("score", ascending=False)

# os.makedirs(os.path.dirname(OUTPUT_IMPORTANCE), exist_ok=True)
# importance_df.to_csv(OUTPUT_IMPORTANCE, index=False)

# print("💾 Saving selected features dataset...")
# final_df = df[selected_features + ["sentiment", "trend"]].copy()
# final_df.to_parquet(OUTPUT_FEATURES, index=False)

# print("✅ Feature engineering completed")
# print("Selected features:", selected_features)
# print(f"📁 Features saved to: {OUTPUT_FEATURES}")
# print(f"📁 Importance saved to: {OUTPUT_IMPORTANCE}")
import pandas as pd

# -------------------------------
# Load OFFLINE parquet data
# -------------------------------
df = pd.read_parquet("data/historical_prices_fixed.parquet")

# -------------------------------
# Clean column names
# -------------------------------
df.columns = df.columns.str.strip()

print("Columns found:", list(df.columns))

# -------------------------------
# FIX: 'Price' IS ACTUALLY DATE
# -------------------------------
if "Price" in df.columns:
    df = df.rename(columns={"Price": "Date"})
else:
    raise ValueError("❌ Expected 'Price' column not found for Date")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# -------------------------------
# FORCE NUMERIC CONVERSION
# -------------------------------
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with invalid Date or Close
df = df.dropna(subset=["Date", "Close"])

# -------------------------------
# Sort data correctly
# -------------------------------
df = df.sort_values(["Company", "Date"])

# -------------------------------
# Feature Engineering
# -------------------------------
df["MA_10"] = df.groupby("Company")["Close"].transform(
    lambda x: x.rolling(10).mean()
)

df["MA_20"] = df.groupby("Company")["Close"].transform(
    lambda x: x.rolling(20).mean()
)

# Remove rows created by rolling window
df = df.dropna()

# -------------------------------
# Save final dataset
# -------------------------------
df.to_parquet("data/master_dataset.parquet", index=False)

print("✅ Feature engineering completed successfully")
print("Final dataset shape:", df.shape)
