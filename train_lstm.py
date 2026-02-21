import os
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# CONFIGURATION
# ==================================================
DATA_PATH = "data/master_dataset.parquet"
SENTIMENT_PATH = "data/news_sentiment.parquet"

LOOKBACK = 50
HORIZON = 3
EPOCHS = 15
BATCH_SIZE = 128
YEARS = 7

BASE_FEATURES = ["Close", "MA_10", "MA_20"]
SENTIMENT_FEATURE = ["Sentiment"]

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================
price_df = pd.read_parquet(DATA_PATH)
price_df["Date"] = pd.to_datetime(price_df["Date"])
price_df = price_df.sort_values(["Company", "Date"])

sent_df = pd.read_parquet(SENTIMENT_PATH)
sent_df["date"] = pd.to_datetime(sent_df["date"])
sent_df = sent_df.rename(columns={"date": "Date", "sentiment": "Sentiment"})

# ==================================================
# FILTER LAST N YEARS (GLOBAL)
# ==================================================
end_date = price_df["Date"].max()
start_date = end_date - timedelta(days=365 * YEARS)
price_df = price_df[price_df["Date"] >= start_date]

# ==================================================
# MERGE SENTIMENT
# ==================================================
price_df = price_df.merge(
    sent_df[["ticker", "Date", "Sentiment"]],
    left_on=["Company", "Date"],
    right_on=["ticker", "Date"],
    how="left"
)

price_df["Sentiment"] = price_df["Sentiment"].fillna(0.0)
price_df.drop(columns=["ticker"], inplace=True)

print(f"📌 Total rows after merge: {len(price_df)}")

# ==================================================
# ENCODE COMPANY AS ID
# ==================================================
le = LabelEncoder()
price_df["company_id"] = le.fit_transform(price_df["Company"])
num_companies = price_df["company_id"].nunique()

# ==================================================
# BUILD SEQUENCES (GLOBAL)
# ==================================================
features = BASE_FEATURES + SENTIMENT_FEATURE
scaler = MinMaxScaler()
price_df[features] = scaler.fit_transform(price_df[features])

X_price, X_company, y = [], [], []

for cid in price_df["company_id"].unique():

    df = price_df[price_df["company_id"] == cid].copy()

    for i in range(LOOKBACK, len(df) - HORIZON):
        X_price.append(df[features].iloc[i - LOOKBACK:i].values)
        X_company.append(cid)
        y.append(df["Close"].iloc[i + HORIZON])

X_price = np.array(X_price)
X_company = np.array(X_company)
y = np.array(y)

print(f"📦 Total sequences built: {len(X_price)}")

# ==================================================
# TRAIN / TEST SPLIT (TIME-AWARE)
# ==================================================
split = int(len(X_price) * 0.8)

X_price_train, X_price_test = X_price[:split], X_price[split:]
X_company_train, X_company_test = X_company[:split], X_company[split:]
y_train, y_test = y[:split], y[split:]

# ==================================================
# MODEL: GLOBAL LSTM + COMPANY EMBEDDING
# ==================================================
price_input = Input(shape=(LOOKBACK, len(features)), name="price_input")
company_input = Input(shape=(1,), name="company_input")

company_emb = Embedding(
    input_dim=num_companies,
    output_dim=8
)(company_input)
company_emb = Dense(8, activation="relu")(company_emb)

x = LSTM(64)(price_input)
x = Dropout(0.2)(x)

x = Concatenate()([x, company_emb[:, 0, :]])
output = Dense(1)(x)

model = Model(
    inputs=[price_input, company_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ==================================================
# TRAIN
# ==================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    [X_price_train, X_company_train],
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ==================================================
# EVALUATE
# ==================================================
y_pred = model.predict([X_price_test, X_company_test], verbose=0).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

directional_accuracy = np.mean(
    np.sign(np.diff(y_test)) ==
    np.sign(np.diff(y_pred))
) * 100

print("\n📊 GLOBAL S&P 500 RESULTS")
print(f"Directional Accuracy (%): {directional_accuracy:.2f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ==================================================
# SAVE RESULTS
# ==================================================
model.save("models/lstm_sp500_global.keras")

pd.DataFrame([{
    "model": "Global LSTM + Sentiment",
    "directional_accuracy_percent": round(directional_accuracy, 2),
    "mae": mae,
    "rmse": rmse
}]).to_csv("data/results_sp500_global_lstm.csv", index=False)

print("\n🎉 GLOBAL S&P 500 LSTM TRAINING COMPLETED")
