import pandas as pd

def show(path):
    print(f"\n=== {path} ===")
    try:
        df = pd.read_parquet(path)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print(df.head())
    except Exception as e:
        print("ERROR reading", path, "->", e)

show("data/historical_prices_fixed.parquet")
show("data/news_sentiment.parquet")
show("data/master_dataset.parquet")
