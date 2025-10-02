from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, concat_ws, collect_list, expr

def create_spark_session():
    """Creates and returns a SparkSession."""
    return SparkSession.builder \
        .appName("Merge Stock Data and News") \
        .getOrCreate()

def main():
    spark = create_spark_session()
    print("SparkSession created successfully.")

    try:
        # Load datasets
        prices_df = spark.read.parquet('data/historical_prices.parquet')
        news_df = spark.read.parquet('data/news_headlines.parquet')
        print("Data loaded into Spark.")

        # Prepare news data: aggregate headlines by ticker and date
        news_df = news_df.withColumn("date", to_date(col("published_at")))
        news_aggregated_df = news_df.groupBy("ticker", "date") \
            .agg(concat_ws(" | ", collect_list("title")).alias("headlines"))
        print("News headlines aggregated by day.")

        # Prepare price data: transform from wide to long format
        stack_cols = ", ".join([f"'{ticker}', `{ticker}`" for ticker in prices_df.columns if ticker != 'Date'])
        prices_long_df = prices_df.select(
            col("Date").alias("date"),
            expr(f"stack({len(prices_df.columns)-1}, {stack_cols}) as (ticker, close)")
        )
        print("Price data transformed to long format.")
        
        # Join the datasets
        master_df = prices_long_df.join(
            news_aggregated_df,
            on=['date', 'ticker'],
            how="left"
        )
        print("Price and news data successfully merged.")
        
        # Save the final dataset
        master_df.write.mode('overwrite').parquet('data/master_dataset.parquet')
        print("Preliminary master dataset saved to 'data/master_dataset.parquet'")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()
        print("SparkSession stopped.")

if __name__ == "__main__":
    main()