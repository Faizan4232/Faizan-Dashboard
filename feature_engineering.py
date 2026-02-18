
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
