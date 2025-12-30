# train_lstm.py
# --------------------------------------------------
# LSTM-based stock trend prediction (paper-aligned)
# --------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = "data/features_selected.parquet"
OUT_NO_SENT = "data/results_without_sentiment.parquet"
OUT_WITH_SENT = "data/results_with_sentiment.parquet"

SEQ_LEN = 20   # time window (T in methodology)
EPOCHS = 20
BATCH_SIZE = 32

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("Run feature_engineering.py first")

df = pd.read_parquet(INPUT_PATH)

# -----------------------------
# Helper: create sequences
# -----------------------------
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

# -----------------------------
# Train function
# -----------------------------
def train_lstm(use_sentiment=True):

    features = [c for c in df.columns if c not in ["trend"]]

    if not use_sentiment and "sentiment" in features:
        features.remove("sentiment")

    X = df[features].values
    y = df["trend"].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create time-series sequences
    X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

    # Temporal train-test split (NO SHUFFLING)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # -----------------------------
    # LSTM Model
    # -----------------------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "directional_accuracy": np.mean(y_test == y_pred),
        "mae": mean_absolute_error(y_test, y_prob),
        "rmse": np.sqrt(mean_squared_error(y_test, y_prob)),
        "use_sentiment": use_sentiment,
    }

    return results

# -----------------------------
# Run experiments
# -----------------------------
os.makedirs("data", exist_ok=True)

pd.DataFrame([train_lstm(False)]).to_parquet(OUT_NO_SENT)
pd.DataFrame([train_lstm(True)]).to_parquet(OUT_WITH_SENT)

print("✅ LSTM experiments completed successfully")
