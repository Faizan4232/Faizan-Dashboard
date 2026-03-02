#ONLY LSTM TRAINING FOR APPL,MSFT,TSLA COMPANIES 



# import os
# import numpy as np
# import pandas as pd
# from datetime import timedelta

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping

# # ==================================================
# # CONFIGURATION
# # ==================================================
# DATA_PATH = "data/master_dataset.parquet"
# SENTIMENT_PATH = "data/news_sentiment.parquet"

# COMPANIES = ["AAPL", "MSFT", "TSLA"]

# LOOKBACK = 50
# HORIZON = 3                 # 🔥 3-day ahead prediction
# EPOCHS = 10
# BATCH_SIZE = 64
# YEARS = 7

# BASE_FEATURES = ["Close", "MA_10", "MA_20"]
# SENTIMENT_FEATURE = ["Sentiment"]

# os.makedirs("models", exist_ok=True)
# os.makedirs("data", exist_ok=True)

# # ==================================================
# # LOAD DATA
# # ==================================================
# price_df = pd.read_parquet(DATA_PATH)
# price_df["Date"] = pd.to_datetime(price_df["Date"])
# price_df = price_df.sort_values("Date")

# sent_df = pd.read_parquet(SENTIMENT_PATH)
# sent_df["date"] = pd.to_datetime(sent_df["date"])

# # ==================================================
# # TRAIN MODELS
# # ==================================================
# for company in COMPANIES:

#     print("\n" + "=" * 70)
#     print(f"🚀 Training models for {company}")
#     print("=" * 70)

#     # -------------------------------
#     # FILTER PRICE DATA
#     # -------------------------------
#     df = price_df[price_df["Company"] == company].copy()

#     end_date = df["Date"].max()
#     start_date = end_date - timedelta(days=365 * YEARS)
#     df = df[df["Date"] >= start_date]

#     # -------------------------------
#     # MERGE SENTIMENT
#     # -------------------------------
#     s = sent_df[sent_df["ticker"] == company][["date", "sentiment"]]
#     s = s.rename(columns={"date": "Date", "sentiment": "Sentiment"})

#     df = df.merge(s, on="Date", how="left")
#     df["Sentiment"] = df["Sentiment"].fillna(0.0)

#     print(f"📅 Data range: {start_date.date()} → {end_date.date()}")

#     # ==================================================
#     # FUNCTION: TRAIN + EVALUATE
#     # ==================================================
#     def train_model(feature_cols, tag):

#         data = df.dropna(subset=feature_cols)

#         scaler = MinMaxScaler()
#         scaled = scaler.fit_transform(data[feature_cols])

#         X, y = [], []
#         for i in range(LOOKBACK, len(scaled) - HORIZON):
#             X.append(scaled[i - LOOKBACK:i])
#             y.append(scaled[i + HORIZON, 0])   # 🔥 3-day ahead Close

#         X = np.array(X)
#         y = np.array(y)

#         split = int(len(X) * 0.8)
#         X_train, X_test = X[:split], X[split:]
#         y_train, y_test = y[:split], y[split:]

#         model = Sequential([
#             Input(shape=(LOOKBACK, X.shape[2])),
#             LSTM(64),
#             Dropout(0.2),
#             Dense(1)
#         ])

#         model.compile(optimizer="adam", loss="mse")

#         early_stop = EarlyStopping(
#             monitor="val_loss",
#             patience=3,
#             restore_best_weights=True
#         )

#         model.fit(
#             X_train,
#             y_train,
#             validation_split=0.1,
#             epochs=EPOCHS,
#             batch_size=BATCH_SIZE,
#             callbacks=[early_stop],
#             verbose=0
#         )

#         y_pred = model.predict(X_test, verbose=0).flatten()

#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#         # 🔹 Directional Accuracy (%)
#         da = np.mean(
#             np.sign(np.diff(y_test)) ==
#             np.sign(np.diff(y_pred))
#         ) * 100

#         model.save(f"models/lstm_{company}_{tag}.keras")

#         return da, mae, rmse

#     # ==================================================
#     # EXPERIMENT 1: WITHOUT SENTIMENT
#     # ==================================================
#     da_ns, mae_ns, rmse_ns = train_model(
#         BASE_FEATURES,
#         "no_sentiment"
#     )

#     # ==================================================
#     # EXPERIMENT 2: WITH SENTIMENT
#     # ==================================================
#     da_s, mae_s, rmse_s = train_model(
#         BASE_FEATURES + SENTIMENT_FEATURE,
#         "with_sentiment"
#     )

#     # ==================================================
#     # SAVE RESULTS (DA IN %)
#     # ==================================================
#     results = pd.DataFrame([
#         {
#             "company": company,
#             "model": "LSTM (No Sentiment)",
#             "directional_accuracy_percent": round(da_ns, 2),
#             "mae": mae_ns,
#             "rmse": rmse_ns
#         },
#         {
#             "company": company,
#             "model": "LSTM + Sentiment",
#             "directional_accuracy_percent": round(da_s, 2),
#             "mae": mae_s,
#             "rmse": rmse_s
#         }
#     ])

#     results.to_parquet(
#         f"data/results_{company}_sentiment.parquet",
#         index=False
#     )

#     # ==================================================
#     # PRINT RESULTS (CLEAN)
#     # ==================================================
#     print(f"\n📊 FINAL RESULTS FOR {company}")
#     print(results)

# print("\n🎉 ALL SENTIMENT EXPERIMENTS COMPLETED SUCCESSFULLY")

#JOURNAL LEVEL S&P500 FULL DATA 
import os
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# CONFIGURATION (CPU FAST MODE)
# ==================================================
DATA_PATH = "data/master_dataset.parquet"
SENTIMENT_PATH = "data/news_sentiment.parquet"

YEARS = 7          # change to 7 for faster experiments
LOOKBACK = 10       # very important for speed
EPOCHS = 20
BATCH_SIZE = 1024   # better for CPU than 2048

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

# ==================================================
# LOAD DATA
# ==================================================
print("📥 Loading dataset...")
df = pd.read_parquet(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

end_date = df["Date"].max()
start_date = end_date - timedelta(days=365 * YEARS)
df = df[df["Date"] >= start_date]

print("Companies:", df["Company"].nunique())

# ==================================================
# MERGE SENTIMENT
# ==================================================
if os.path.exists(SENTIMENT_PATH):
    sent = pd.read_parquet(SENTIMENT_PATH)
    sent["date"] = pd.to_datetime(sent["date"])
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
# FEATURE ENGINEERING
# ==================================================
df = df.sort_values(["Company", "Date"])

df["log_return"] = np.log(
    df["Close"] / df.groupby("Company")["Close"].shift(1)
)

df["Volatility_5"] = (
    df.groupby("Company")["log_return"]
    .rolling(5)
    .std()
    .reset_index(level=0, drop=True)
)

df["Momentum_3"] = df.groupby("Company")["Close"].pct_change(3)

if "Volume" in df.columns:
    df["Volume_Change"] = df.groupby("Company")["Volume"].pct_change()
else:
    df["Volume_Change"] = 0

df["Direction"] = (
    df.groupby("Company")["log_return"]
    .shift(-1) > 0
).astype(int)

# Clean bad values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

FEATURES = [
    "Close",
    "Sentiment",
    "Volatility_5",
    "Momentum_3",
    "Volume_Change"
]

FEATURES = [f for f in FEATURES if f in df.columns]

# ==================================================
# TIME SPLIT
# ==================================================
split_date = df["Date"].quantile(0.8)

train_df = df[df["Date"] <= split_date].copy()
test_df = df[df["Date"] > split_date].copy()

# ==================================================
# SCALE
# ==================================================
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# ==================================================
# SEQUENCE CREATION (FAST NUMPY)
# ==================================================
def create_sequences(data):
    X, y = [], []

    for _, group in data.groupby("Company"):
        values = group[FEATURES + ["Direction"]].values
        if len(values) < LOOKBACK:
            continue

        for i in range(LOOKBACK, len(values)):
            X.append(values[i-LOOKBACK:i, :-1])
            y.append(values[i, -1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

print("⚙ Creating sequences...")
X_train, y_train = create_sequences(train_df)
X_test, y_test = create_sequences(test_df)

print("Train sequences:", len(X_train))
print("Test sequences:", len(X_test))

# ==================================================
# SIMPLE FAST LSTM (BEST FOR CPU)
# ==================================================
model = Sequential([
    LSTM(32, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ==================================================
# TRAIN
# ==================================================
print("🚀 Training...")

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
print("📊 Evaluating...")

y_prob = model.predict(X_test, batch_size=2048).flatten()
y_pred = (y_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

results = pd.DataFrame([{
    "Model": "CPU Fast LSTM - 10Y S&P500",
    "Accuracy_%": round(accuracy * 100, 2),
    "ROC_AUC": round(auc, 4),
    "F1_Score": round(f1, 4),
    "MCC": round(mcc, 4)
}])

results.to_csv("data/sp500_cpu_fast_lstm_results.csv", index=False)
model.save("models/sp500_cpu_fast_lstm.keras")

print("\n📊 FINAL RESULTS")
print(results)
print("\n✅ TRAINING COMPLETE")