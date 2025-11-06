"""
merge_data.py - Merge S&P 500 price (wide) and news (long) Parquet data with PySpark
Usage:
    python merge_data.py --prices data/historical_prices.parquet \
                        --news data/news_headlines.parquet \
                        --output data/master_dataset.parquet
Requires: pyspark
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, concat_ws, collect_list, expr

def create_spark_session():
    return SparkSession.builder \
        .appName("Merge Stock Data and News") \
        .getOrCreate()

def file_exists(path):
    return os.path.exists(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', default='data/historical_prices.parquet', help='Input prices Parquet file (wide format, columns: Date, TICKER1, TICKER2, ...)')
    parser.add_argument('--news', default='data/news_headlines.parquet', help='Input news Parquet file (columns: ticker, published_at, title)')
    parser.add_argument('--output', default='data/master_dataset.parquet', help='Output merged Parquet file')
    args = parser.parse_args()

    if not file_exists(args.prices):
        print(f"Prices file not found: {args.prices}")
        return
    if not file_exists(args.news):
        print(f"News file not found: {args.news}")
        return
    spark = create_spark_session()
    try:
        print("Loading price data...")
        prices_df = spark.read.parquet(args.prices)
        print("Loading news data...")
        news_df = spark.read.parquet(args.news)
        print(f"Price columns: {prices_df.columns}")

        # Aggregate news by ticker and date
        if not ("ticker" in news_df.columns and "published_at" in news_df.columns and "title" in news_df.columns):
            print("News parquet missing required columns: ticker, published_at, title")
            return
        news_df = news_df.withColumn("date", to_date(col("published_at")))
        news_aggregated_df = news_df.groupBy("ticker", "date") \
            .agg(concat_ws(" | ", collect_list("title")).alias("headlines"))
        print(f"Aggregated news shape: {news_aggregated_df.count()}")

        # Unpivot prices from wide to long format
        price_cols = [c for c in prices_df.columns if c != "Date"]
        stack_expr = ", ".join([f"'{ticker}', `{ticker}`" for ticker in price_cols])
        prices_long_df = prices_df.select(
            col("Date").alias("date"),
            expr(f"stack({len(price_cols)}, {stack_expr}) as (ticker, close)")
        )
        print(f"Prices long format shape: {prices_long_df.count()}")

        # Join price and aggregated news
        master_df = prices_long_df.join(
            news_aggregated_df,
            on=['date', 'ticker'],
            how="left"
        )
        print(f"Merged dataset shape: {master_df.count()}")

        # Save merged output
        outdir = os.path.dirname(args.output) or '.'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        master_df.write.mode('overwrite').parquet(args.output)
        print(f"Merged dataset saved to {args.output}.")
    except Exception as e:
        print(f"Error during merge: {e}")
    finally:
        spark.stop()
        print("SparkSession stopped.")

if __name__ == "__main__":
    main()
