"""
feature_engineering.py
----------------------
Spark-based feature engineering for stock market trend prediction.

Input : data/master_dataset.parquet
Output: data/features_dataset.parquet

Features:
- Technical indicators
- Daily sentiment
- Trend label
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------
# Spark Session
# -------------------------------
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .getOrCreate()

# -------------------------------
# Load merged dataset
# -------------------------------
df = spark.read.parquet("data/master_dataset.parquet")
pdf = df.toPandas().dropna()

# -------------------------------
# Technical Indicators
# -------------------------------
pdf["EMA_20"] = trend.EMAIndicator(pdf["close"], window=20).ema_indicator()
pdf["RSI_14"] = momentum.RSIIndicator(pdf["close"], window=14).rsi()
pdf["MACD"] = trend.MACD(pdf["close"]).macd()
pdf["BB_width"] = (
    volatility.BollingerBands(pdf["close"]).bollinger_hband() -
    volatility.BollingerBands(pdf["close"]).bollinger_lband()
)

# -------------------------------
# Trend Label (MANDATORY)
# -------------------------------
pdf["Trend"] = (pdf["close"].shift(-1) > pdf["close"]).astype(int)
pdf.dropna(inplace=True)

# -------------------------------
# Feature Selection
# -------------------------------
feature_cols = ["EMA_20", "RSI_14", "MACD", "BB_width", "daily_sentiment"]
X = pdf[feature_cols]
y = pdf["Trend"]

selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))

final_df = pdf[list(selected_features) + ["Trend"]]

# -------------------------------
# Save for LSTM
# -------------------------------
final_df.to_parquet("data/features_dataset.parquet", index=False)
print("Feature dataset saved: data/features_dataset.parquet")

spark.stop()
