"""
feature_engineering.py
----------------------
Spark-based feature engineering and feature selection for
stock market trend prediction.

Input  : data/master_dataset.parquet
Output :
    - data/features_dataset.parquet
    - data/feature_importance.csv

Implements:
- Technical indicators
- Trend labeling
- Feature selection (SelectKBest)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, when
from pyspark.sql.window import Window
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

# -----------------------------
# Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("FeatureEngineeringStockPrediction") \
    .getOrCreate()

# -----------------------------
# Load merged dataset
# -----------------------------
df = spark.read.parquet("data/master_dataset.parquet")

# Ensure required columns
required_cols = ["date", "ticker", "close"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# -----------------------------
# Window specification
# -----------------------------
w = Window.partitionBy("ticker").orderBy("date")

# -----------------------------
# Technical Indicators
# -----------------------------
# Simple Moving Average
df = df.withColumn("SMA_5", lag("close", 5).over(w))
df = df.withColumn("SMA_10", lag("close", 10).over(w))

# Exponential Moving Average (approx)
df = df.withColumn("EMA_5", lag("close", 5).over(w))
df = df.withColumn("EMA_10", lag("close", 10).over(w))

# Momentum
df = df.withColumn("Momentum", col("close") - lag("close", 1).over(w))

# Volatility (rolling std approximation)
df = df.withColumn("Volatility", col("close") - lag("close", 5).over(w))

# -----------------------------
# Trend Label (TARGET)
# -----------------------------
df = df.withColumn(
    "Trend",
    when(lag("close", -1).over(w) > col("close"), 1).otherwise(0)
)

# -----------------------------
# Select features for ML
# -----------------------------
feature_cols = [
    "SMA_5", "SMA_10",
    "EMA_5", "EMA_10",
    "Momentum", "Volatility"
]

# Optional sentiment (if present)
if "daily_sentiment" in df.columns:
    feature_cols.append("daily_sentiment")

# Drop null rows
df_ml = df.select(["date", "ticker", "Trend"] + feature_cols).dropna()

# Convert to Pandas for feature selection
pdf = df_ml.toPandas()

X = pdf[feature_cols].values
y = pdf["Trend"].values

# -----------------------------
# Feature Selection
# -----------------------------
selector = SelectKBest(score_func=f_classif, k=min(6, X.shape[1]))
X_selected = selector.fit_transform(X, y)

selected_features = [
    feature_cols[i] for i in selector.get_support(indices=True)
]

# Save feature importance
feature_scores = pd.DataFrame({
    "Feature": feature_cols,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False)

feature_scores.to_csv("data/feature_importance.csv", index=False)

# -----------------------------
# Save final dataset
# -----------------------------
final_df = pdf[["date", "ticker", "Trend"] + selected_features]
final_df.to_parquet("data/features_dataset.parquet", index=False)

print("Feature engineering completed.")
print("Selected features:", selected_features)

spark.stop()
