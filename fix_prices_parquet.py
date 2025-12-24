# fix_prices_parquet.py
# Converts Yahoo-style MultiIndex OHLCV parquet into Spark-ready long format

import pandas as pd
import os

INPUT_PATH = "data/historical_prices.parquet"
OUTPUT_PATH = "data/historical_prices_fixed.parquet"

print("📥 Loading raw prices parquet...")
df = pd.read_parquet(INPUT_PATH)

# --------------------------------------------------
# STEP 1: Ensure 'date' column exists
# --------------------------------------------------
if isinstance(df.index, pd.DatetimeIndex):
    print("✅ Date found in index, resetting index...")
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

# If date is already a column with another name, standardize it
if "date" not in df.columns:
    # Try common variants
    for cand in ["Date", "timestamp", "Datetime"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "date"})
            break

if "date" not in df.columns:
    raise RuntimeError("❌ Could not find a date column in the DataFrame.")

# --------------------------------------------------
# STEP 2: Confirm MultiIndex columns
# --------------------------------------------------
if not isinstance(df.columns, pd.MultiIndex):
    raise RuntimeError("❌ Expected MultiIndex columns (ticker, OHLCV).")

print("🔧 Processing MultiIndex OHLCV structure...")
print("Columns example:", df.columns[:6])

# --------------------------------------------------
# STEP 3: Convert wide → long format
# --------------------------------------------------
records = []

# Level names are often like ('AAPL','Open'), ('AAPL','High'), etc.
# Get all tickers from level 0 except any non-ticker like 'date'
tickers = [t for t in df.columns.levels[0] if t != "date"]

for ticker in tickers:
    # Check which fields exist for this ticker
    available_fields = df[ticker].columns.tolist()
    needed = ["Open", "High", "Low", "Close", "Volume"]

    if not all(f in available_fields for f in needed):
        # Skip tickers with incomplete OHLCV
        continue

    # Build temp DataFrame for this ticker
    temp = pd.DataFrame()
    temp["date"] = df["date"]  # simple 1D column, not MultiIndex
    temp["open"] = df[(ticker, "Open")]
    temp["high"] = df[(ticker, "High")]
    temp["low"] = df[(ticker, "Low")]
    temp["close"] = df[(ticker, "Close")]
    temp["volume"] = df[(ticker, "Volume")]
    temp["ticker"] = ticker

    records.append(temp)

# --------------------------------------------------
# STEP 4: Combine all tickers
# --------------------------------------------------
if not records:
    raise RuntimeError(
        "❌ No ticker OHLCV data found. "
        "Check that your MultiIndex is (ticker, field) and that OHLCV columns exist."
    )

final_df = pd.concat(records, ignore_index=True)

# --------------------------------------------------
# STEP 5: Clean and save
# --------------------------------------------------
final_df["date"] = pd.to_datetime(final_df["date"]).dt.date.astype(str)
final_df.dropna(inplace=True)

os.makedirs("data", exist_ok=True)
final_df.to_parquet(OUTPUT_PATH, index=False)

print("✅ Prices fixed successfully!")
print(f"📁 Saved to: {OUTPUT_PATH}")
print("📊 Final columns:", list(final_df.columns))
print("🔢 Rows:", len(final_df))
