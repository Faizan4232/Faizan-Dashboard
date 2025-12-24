# merge_data.py
"""
Merge historical stock prices with daily news sentiment (Spark for join, pandas for save).

Inputs:
  data/historical_prices_fixed.parquet  (date, open, high, low, close, volume, ticker)
  data/news_sentiment.parquet           (ticker, date, sentiment)

Output:
  data/master_dataset.parquet
"""

import os
import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


def main(prices_path, news_path, output_path):
    spark = (
        SparkSession.builder
        .appName("MergeStockAndSentiment")
        .getOrCreate()
    )

    try:
        print("📥 Loading price data from:", prices_path)
        prices_df = spark.read.parquet(prices_path)
        print("Price schema:")
        prices_df.printSchema()
        print("Price count:", prices_df.count())

        # Ensure date is proper date
        if "date" in prices_df.columns:
            prices_df = prices_df.withColumn(
                "date", to_date(col("date").cast("string"))
            )
        elif "Date" in prices_df.columns:
            prices_df = prices_df.withColumn(
                "date", to_date(col("Date").cast("string"))
            )
        else:
            raise ValueError("Price data must contain 'date' or 'Date' column")

        prices_df = prices_df.select(
            col("ticker"),
            col("date"),
            col("open"),
            col("high"),
            col("low"),
            col("close"),
            col("volume"),
        )

        print("📥 Loading news sentiment data from:", news_path)
        news_df = spark.read.parquet(news_path)
        print("News schema:")
        news_df.printSchema()
        print("News count:", news_df.count())

        news_df = news_df.withColumn(
            "date", to_date(col("date").cast("string"))
        )

        news_df = news_df.select(
            col("ticker"),
            col("date"),
            col("sentiment").alias("daily_sentiment"),
        )

        print("🔗 Merging datasets (left join on ticker,date)...")
        merged_df = prices_df.join(
            news_df,
            on=["ticker", "date"],
            how="left",
        )

        print("Merged count:", merged_df.count())
        merged_df = merged_df.fillna({"daily_sentiment": 0.0})

        # Convert to pandas and save to avoid Hadoop Windows native IO issues
        print("💾 Converting merged Spark DataFrame to pandas...")
        merged_pd = merged_df.toPandas()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_pd.to_parquet(output_path, index=False)

        print(f"✅ Master dataset saved to {output_path} via pandas")
        print("Merged pandas shape:", merged_pd.shape)

    except Exception as e:
        print(f"❌ Error during merge: {e}")

    finally:
        spark.stop()
        print("🛑 Spark session stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices",
        default="data/historical_prices_fixed.parquet",
        help="Path to Spark-safe historical OHLCV parquet file",
    )
    parser.add_argument(
        "--news",
        default="data/news_sentiment.parquet",
        help="Path to daily news sentiment parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/master_dataset.parquet",
        help="Output merged parquet path",
    )
    args = parser.parse_args()
    main(args.prices, args.news, args.output)
