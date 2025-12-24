"""
train_lstm.py
---------------------------------
LSTM-based stock market trend prediction
Supports experiments:
1) Without sentiment
2) With sentiment

Input:
- data/features_dataset.parquet   (from feature_engineering.py)
Output:
- data/results_without_sentiment.parquet
- data/results_with_sentiment.parquet
"""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Spark Session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("LSTM Stock Trend Prediction") \
    .getOrCreate()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
FEATURES_FILE = "data/features_dataset.parquet"
RESULTS_NO_SENT = "data/results_without_sentiment.parquet"
RESULTS_WITH_SENT = "data/results_with_sentiment.parquet"

TIME_STEPS = 10
EPOCHS = 20
BATCH_SIZE = 32

# --------------------------------------------------
# Load Feature Dataset (Spark → Pandas)
# --------------------------------------------------
print("Loading feature dataset...")
spark_df = spark.read.parquet(FEATURES_FILE)
df = spark_df.toPandas()

# Sort by date for time-series integrity
df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

# --------------------------------------------------
# Trend Label (MANDATORY for paper)
# --------------------------------------------------
df["Trend"] = (df["close"].shift(-1) > df["close"]).astype(int)
df.dropna(inplace=True)

# --------------------------------------------------
# Feature Sets
# --------------------------------------------------
indicator_cols = [
    col for col in df.columns
    if col not in ["date", "ticker", "Trend", "daily_sentiment"]
]

indicator_plus_sentiment_cols = indicator_cols + ["daily_sentiment"]

# --------------------------------------------------
# Helper: Create Sequences for LSTM
# --------------------------------------------------
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --------------------------------------------------
# Helper: Train & Evaluate LSTM
# --------------------------------------------------
def train_and_evaluate(features, result_path):
    print(f"\nTraining LSTM using features: {features}")

    X = df[features].values
    y = df["Trend"].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

    # Train / Test split (time-based)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # LSTM Model
    model = Sequential([
        LSTM(50, input_shape=(TIME_STEPS, X_train.shape[2])),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Metrics (conference-required)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    directional_accuracy = np.mean(y_test == y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f}")

    # Save results
    results_df = pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "Directional_Accuracy": [directional_accuracy]
    })

    results_df.to_parquet(result_path, index=False)
    print(f"Results saved to {result_path}")

# --------------------------------------------------
# Experiment 1: WITHOUT Sentiment
# --------------------------------------------------
train_and_evaluate(
    features=indicator_cols,
    result_path=RESULTS_NO_SENT
)

# --------------------------------------------------
# Experiment 2: WITH Sentiment
# --------------------------------------------------
if "daily_sentiment" in df.columns:
    train_and_evaluate(
        features=indicator_plus_sentiment_cols,
        result_path=RESULTS_WITH_SENT
    )
else:
    print("Sentiment column not found. Skipping sentiment experiment.")

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
spark.stop()
print("Training complete.")
