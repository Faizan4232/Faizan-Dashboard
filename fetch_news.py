"""
fetch_news.py - Fetch and save news-based sentiment for S&P 500 companies.

Usage:

python fetch_news.py --apikey YOUR_API_KEY --output data/news_sentiment.parquet --max_articles 10

Requires: pandas, requests, newsapi-python, tqdm, textblob
"""

import pandas as pd
import os
import requests
import argparse

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from newsapi import NewsApiClient
except ImportError:
    raise ImportError("Please install 'newsapi-python' via 'pip install newsapi-python'")

from textblob import TextBlob  # sentiment analysis

# --- Default Config ---

# Do NOT hardcode real keys in code shown in a paper
DEFAULT_API_KEY = None  # use --apikey argument


def compute_sentiment(text):
    """Compute simple polarity score in [-1, 1] using TextBlob."""
    if not text or pd.isna(text):
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def get_sp500_companies():
    """Scrapes S&P 500 company names and tickers from Wikipedia."""
    print("Fetching S&P 500 company list...")

    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    response = requests.get(sp500_url, headers=headers)
    response.raise_for_status()

    sp500_df = pd.read_html(response.text, header=0)[0]
    companies = sp500_df[["Symbol", "Security"]].to_dict("records")

    for company in companies:
        company["Symbol"] = company["Symbol"].replace(".", "-")

    print(f"Loaded {len(companies)} company names and tickers.")
    return companies


def fetch_all_news(companies, apikey, max_articles=10):
    """Fetches news articles for a list of companies using NewsAPI and computes sentiment per headline."""
    newsapi = NewsApiClient(api_key=apikey)
    all_articles = []

    print(f"\nFetching news articles for up to {max_articles} per company...")
    iterator = tqdm(companies, desc="Companies") if tqdm else companies

    for company in iterator:
        ticker = company["Symbol"]
        company_name = company["Security"]

        try:
            articles = newsapi.get_everything(
                q=f'"{company_name}"',
                language="en",
                sort_by="publishedAt",
                page_size=max_articles,  # Up to 100 per API, use small number for quota
            )

            for article in articles.get("articles", []):
                title = article.get("title")
                all_articles.append(
                    {
                        "ticker": ticker,
                        "published_at": article.get("publishedAt"),
                        "title": title,
                        "sentiment": compute_sentiment(title),
                    }
                )
        except Exception as e:
            print(f"Could not fetch news for {company_name} ({ticker}): {e}")

    news_df = pd.DataFrame(all_articles)
    return news_df


def save_news_data(df, path="data/news_sentiment.parquet"):
    """Aggregates sentiment daily and saves the result as Parquet."""
    if df is None or df.empty:
        print("\nNo news articles fetched. Nothing to save.")
        return None

    df["published_at"] = pd.to_datetime(df["published_at"])
    df["date"] = df["published_at"].dt.date

    # Aggregate daily average sentiment per ticker
    daily_sentiment = (
        df.groupby(["ticker", "date"])["sentiment"]
        .mean()
        .reset_index()
    )

    outdir = os.path.dirname(path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    daily_sentiment.to_parquet(path)

    print(f"\nSuccessfully fetched {len(df)} raw articles.")
    print(f"Daily sentiment rows: {len(daily_sentiment)}")
    print(f"News sentiment data saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", default=DEFAULT_API_KEY, help="NewsAPI key. Required.")
    parser.add_argument(
        "--output",
        default="data/news_sentiment.parquet",
        help="Output path for daily sentiment parquet.",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=10,
        help="Max articles per company [1-100].",
    )

    args = parser.parse_args()

    if not args.apikey:
        print("Error: Please provide a valid NewsAPI key using --apikey.")
        return

    company_list = get_sp500_companies()
    if company_list:
        news_data = fetch_all_news(
            company_list,
            apikey=args.apikey,
            max_articles=args.max_articles,
        )
        save_news_data(news_data, path=args.output)


if __name__ == "__main__":
    main()
