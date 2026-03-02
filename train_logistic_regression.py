# ==================================================
# SIMPLE LOGISTIC REGRESSION BASELINE
# (INTENTIONALLY SIMPLE — FAIR COMPARISON)
# ==================================================

import os
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# ==================================================
# CONFIGURATION
# ==================================================
DATA_PATH = "data/master_dataset.parquet"

COMPANIES = ["AAPL", "MSFT", "TSLA"]

HORIZON = 5
YEARS = 7

# Keep it simple (no advanced features)
BASE_FEATURES = ["Return_1", "MA_10", "MA_20"]
SENTIMENT_FEATURE = ["Sentiment"]

# ==================================================
# LOAD DATA
# ==================================================
df_all = pd.read_parquet(DATA_PATH)
df_all["Date"] = pd.to_datetime(df_all["Date"])
df_all = df_all.sort_values("Date")

print("Available columns:", list(df_all.columns))

# ==================================================
# TRAIN LOOP
# ==================================================
for company in COMPANIES:

    print("\n" + "=" * 70)
    print(f"📊 Simple Logistic Baseline for {company}")
    print("=" * 70)

    df = df_all[df_all["Company"] == company].copy()

    end_date = df["Date"].max()
    start_date = end_date - timedelta(days=365 * YEARS)
    df = df[df["Date"] >= start_date]

    # --------------------------------------------------
    # Directional Target (simple)
    # --------------------------------------------------
    df["Future_Return"] = df["Return_1"].shift(-HORIZON)
    df["Target"] = (df["Future_Return"] > 0).astype(int)

    df = df.dropna()

    print(f"📅 Data range: {start_date.date()} → {end_date.date()}")

    def train_logistic(feature_cols):

        data = df.dropna(subset=feature_cols + ["Target"]).copy()

        X = data[feature_cols].values
        y = data["Target"].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        y_pred_class = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        da = accuracy_score(y_test, y_pred_class) * 100
        mae = mean_absolute_error(y_test, y_pred_prob)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))

        return da, mae, rmse

    # WITHOUT SENTIMENT
    da_ns, mae_ns, rmse_ns = train_logistic(BASE_FEATURES)

    # WITH SENTIMENT
    if "Sentiment" in df.columns:
        da_s, mae_s, rmse_s = train_logistic(BASE_FEATURES + SENTIMENT_FEATURE)
    else:
        da_s, mae_s, rmse_s = None, None, None

    # Print Results
    results = pd.DataFrame([
        {
            "Company": company,
            "Model": "Logistic (No Sentiment)",
            "DA (%)": round(da_ns, 2),
            "MAE": round(mae_ns, 6),
            "RMSE": round(rmse_ns, 6)
        }
    ])

    if da_s is not None:
        results.loc[len(results)] = [
            company,
            "Logistic + Sentiment",
            round(da_s, 2),
            round(mae_s, 6),
            round(rmse_s, 6)
        ]

    print("\n📊 BASELINE RESULTS")
    print(results)

print("\n🎉 SIMPLE LOGISTIC BASELINE COMPLETED")