"""
train_lstm.py
-------------
LSTM-based stock market trend prediction.

Input : data/features_dataset.parquet
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_parquet("data/features_dataset.parquet")

X = df.drop(columns=["Trend"]).values
y = df["Trend"].values

# -------------------------------
# Scaling
# -------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sequence Creation
# -------------------------------
def create_sequences(X, y, window=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y)

# -------------------------------
# Train-Test Split
# -------------------------------
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# -------------------------------
# LSTM Model
# -------------------------------
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# Training
# -------------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# -------------------------------
# Evaluation
# -------------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
directional_accuracy = np.mean(y_test == y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("Directional Accuracy:", directional_accuracy)
