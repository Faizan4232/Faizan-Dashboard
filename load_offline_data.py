from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, to_date

def load_offline_stock_data(
    data_path="data/raw/archive5/*.csv"
):
    spark = SparkSession.builder \
        .appName("OfflineStockLoader") \
        .getOrCreate()

    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(data_path)

    # Extract company ticker from filename
    df = df.withColumn(
        "Company",
        regexp_extract(input_file_name(), r"([^/]+)\.csv$", 1)
    )

    # Keep required columns
    df = df.select(
        "Company", "Date", "Open", "High", "Low", "Close", "Volume"
    )

    # Convert Date to date type
    df = df.withColumn("Date", to_date("Date"))

    # Drop invalid rows
    df = df.dropna(subset=["Date", "Close"])

    return df
