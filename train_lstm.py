import os
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# CONFIGURATION (FAST + JOURNAL SAFE)
# ==================================================
DATA_PATH = "data/master_dataset.parquet"
SENTIMENT_PATH = "data/news_sentiment.parquet"

LOOKBACK = 15          # 🔥 reduced (major speed gain)
HORIZON = 1
EPOCHS = 30
BATCH_SIZE = 512       # 🔥 fewer steps per epoch
YEARS = 10

BASE_FEATURES = ["Close"]
OPTIONAL_FEATURES = ["MA_10", "MA_20"]

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

end_date = df["Date"].max()
start_date = end_date - timedelta(days=365 * YEARS)
df = df[df["Date"] >= start_date]

# ==================================================
# MERGE SENTIMENT (SAFE)
# ==================================================
if os.path.exists(SENTIMENT_PATH):
    sent = pd.read_parquet(SENTIMENT_PATH)
    sent["date"] = pd.to_datetime(sent["date"], errors="coerce")
    sent = sent.rename(columns={"date": "Date", "sentiment": "Sentiment"})

    df = df.merge(
        sent[["ticker", "Date", "Sentiment"]],
        left_on=["Company", "Date"],
        right_on=["ticker", "Date"],
        how="left"
    )
    df["Sentiment"] = df["Sentiment"].fillna(0.0)
    df.drop(columns=["ticker"], inplace=True)
else:
    df["Sentiment"] = 0.0

# ==================================================
# FEATURE SELECTION
# ==================================================
FEATURES = []

for c in BASE_FEATURES:
    if c in df.columns:
        FEATURES.append(c)

for c in OPTIONAL_FEATURES:
    if c in df.columns:
        FEATURES.append(c)

FEATURES.append("Sentiment")

print(f"📌 Features used: {FEATURES}")

# ==================================================
# SORT + LOG RETURNS
# ==================================================
df = df.sort_values(["Company", "Date"])
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna(subset=["log_return"])

# ==================================================
# SCALE ONCE (GLOBAL)
# ==================================================
scaler = StandardScaler()
df[FEATURES + ["log_return"]] = scaler.fit_transform(
    df[FEATURES + ["log_return"]]
)

# ==================================================
# BUILD SEQUENCES (FAST, NUMPY-ONLY)
# ==================================================
X, y = [], []
skipped = 0

for company in df["Company"].unique():
    sub = df[df["Company"] == company]
    if len(sub) < LOOKBACK + HORIZON:
        skipped += 1
        continue

    vals = sub[FEATURES + ["log_return"]].values

    for i in range(LOOKBACK, len(vals) - HORIZON):
        X.append(vals[i - LOOKBACK:i, :-1])
        y.append(vals[i + HORIZON, -1])

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

print(f"📦 Sequences: {len(X)} | Skipped companies: {skipped}")

# ==================================================
# TRAIN / TEST SPLIT (TIME-AWARE)
# ==================================================
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# FAST LSTM MODEL (NO ATTENTION)
# ==================================================
model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, X.shape[2])),
    Dropout(0.25),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

# ==================================================
# TRAIN
# ==================================================
early_stop = EarlyStopping(
    patience=6,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ==================================================
# EVALUATION
# ==================================================
y_pred = model.predict(X_test, batch_size=1024).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
directional_accuracy = (
    np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
)

# ==================================================
# SAVE RESULTS
# ==================================================
results = pd.DataFrame([{
    "Model": "Fast LSTM (Journal Optimized)",
    "MAE": round(mae, 4),
    "MAPE": round(mape, 4),
    "RMSE": round(rmse, 4),
    "Directional_Accuracy_%": round(directional_accuracy, 2)
}])

results.to_csv("data/fast_journal_results.csv", index=False)
model.save("models/fast_journal_lstm.keras")

print("\n📊 FINAL FAST RESULTS")
print(results)
print("\n✅ TRAINING FINISHED (FAST MODE)")