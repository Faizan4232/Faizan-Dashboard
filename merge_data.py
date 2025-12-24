"""
merge_data.py - Merge S&P 500 OHLCV prices and daily news sentiment with PySpark.

Usage:
    python merge_data.py --prices data/historical_ohlcv.parquet \
        --news data/news_sentiment.parquet \
        --output data/master_dataset.parquet

Requires:
    pyspark
"""

# Spark-based distributed merge for large-scale financial and news/sentiment data

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, expr


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Merge Stock OHLCV and Sentiment")
        .getOrCreate()
    )


def file_exists(path):
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices",
        default="data/historical_ohlcv.parquet",
        help="Input OHLCV Parquet file (MultiIndex columns flattened by Spark).",
    )
    parser.add_argument(
        "--news",
        default="data/news_sentiment.parquet",
        help="Input daily sentiment Parquet (columns: ticker, date, sentiment).",
    )
    parser.add_argument(
        "--output",
        default="data/master_dataset.parquet",
        help="Output merged Parquet file.",
    )

    args = parser.parse_args()

    if not file_exists(args.prices):
        print(f"Prices file not found: {args.prices}")
        return

    if not file_exists(args.news):
        print(f"News sentiment file not found: {args.news}")
        return

    spark = create_spark_session()

    try:
        print("Loading price data...")
        prices_df = spark.read.parquet(args.prices)

        print("Loading sentiment data...")
        news_df = spark.read.parquet(args.news)

        # Expect: news_df has columns: ticker, date, sentiment
        if not (
            "ticker" in news_df.columns
            and "date" in news_df.columns
            and "sentiment" in news_df.columns
        ):
            print("News parquet missing required columns: ticker, date, sentiment")
            return

        # If OHLCV MultiIndex was flattened, you may have columns like:
        # ('AAPL','Close') -> 'AAPL_Close'. For simplicity here, assume a wide Close matrix
        # If your prices_df is wide with 'Date' + tickers as Close,
        # unpivot (stack) to long format: (date, ticker, close)
        print(f"Price columns: {prices_df.columns}")

        if "Date" not in prices_df.columns:
            print("Expected 'Date' column in prices parquet.")
            return

        price_cols = [c for c in prices_df.columns if c != "Date"]
        if not price_cols:
            print("No ticker columns found in prices parquet.")
            return

        stack_expr = ", ".join([f"'{ticker}', `{ticker}`" for ticker in price_cols])

        prices_long_df = prices_df.select(
            col("Date").alias("date"),
            expr(f"stack({len(price_cols)}, {stack_expr}) as (ticker, close)"),
        )

        print(f"Prices long format count: {prices_long_df.count()}")

        # Join prices with sentiment on (date, ticker)
        master_df = prices_long_df.join(
            news_df,
            on=["date", "ticker"],
            how="left",
        )

        print(f"Merged dataset count: {master_df.count()}")

        # Save merged output
        outdir = os.path.dirname(args.output) or "."
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        master_df.write.mode("overwrite").parquet(args.output)
        print(f"Merged dataset saved to {args.output}.")

    except Exception as e:
        print(f"Error during merge: {e}")

    finally:
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    main()
