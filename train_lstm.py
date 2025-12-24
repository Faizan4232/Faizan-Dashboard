# train_lstm.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

INPUT_PATH = "data/features_selected.parquet"
OUT_NO_SENT = "data/results_without_sentiment.parquet"
OUT_WITH_SENT = "data/results_with_sentiment.parquet"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("features_selected.parquet not found. Run feature_engineering.py")

df = pd.read_parquet(INPUT_PATH)

def train_lstm(use_sentiment=True):
    features = [c for c in df.columns if c not in ["trend"]]
    if not use_sentiment and "sentiment" in features:
        features.remove("sentiment")

    X = df[features].values
    y = df["trend"].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = X.reshape(X.shape[0], 1, X.shape[1])

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, input_shape=(1, X.shape[2])),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    return {
        "MAE": mean_absolute_error(y_test, y_pred_prob),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_prob)),
        "Directional_Accuracy": np.mean(y_test == y_pred),
        "use_sentiment": use_sentiment,
    }

os.makedirs("data", exist_ok=True)
pd.DataFrame([train_lstm(False)]).to_parquet(OUT_NO_SENT)
pd.DataFrame([train_lstm(True)]).to_parquet(OUT_WITH_SENT)

print("✅ LSTM experiments completed")
