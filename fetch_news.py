"""
fetch.py - Fetch and save news headlines for S&P 500 companies.
Usage:
    python fetch.py --apikey YOUR_API_KEY --output data/news_headlines.parquet --max_articles 10
Requires: pandas, requests, newsapi-python, tqdm
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

# --- Default Config ---
DEFAULT_API_KEY = 'f260d5675ae449d5a1e6a30ea49a6410' # Change to your real key or use --apikey argument


def get_sp500_companies():
    """Scrapes S&P 500 company names and tickers from Wikipedia."""
    print("Fetching S&P 500 company list...")
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(sp500_url, headers=headers)
    sp500_df = pd.read_html(response.text, header=0)[0]
    companies = sp500_df[['Symbol', 'Security']].to_dict('records')
    for company in companies:
        company['Symbol'] = company['Symbol'].replace('.', '-')
    print(f"Loaded {len(companies)} company names and tickers.")
    return companies


def fetch_all_news(companies, apikey, max_articles=10):
    """Fetches news articles for a list of companies using NewsAPI."""
    newsapi = NewsApiClient(api_key=apikey)
    all_articles = []
    print(f"\nFetching news articles for up to {max_articles} per company...")
    iterator = tqdm(companies, desc='Companies') if tqdm else companies
    for company in iterator:
        ticker = company['Symbol']
        company_name = company['Security']
        try:
            articles = newsapi.get_everything(
                q=f'"{company_name}"',
                language='en',
                sort_by='publishedAt',
                page_size=max_articles  # Up to 100 per API, we use small number for quota
            )
            for article in articles['articles']:
                all_articles.append({
                    'ticker': ticker,
                    'published_at': article.get('publishedAt'),
                    'title': article.get('title')
                })
        except Exception as e:
            print(f"Could not fetch news for {company_name} ({ticker}): {e}")
    news_df = pd.DataFrame(all_articles)
    return news_df


def save_news_data(df, path='data/news_headlines.parquet'):
    """Saves the news DataFrame to a Parquet file."""
    if df is None or df.empty:
        print("\nNo news articles fetched. Nothing to save.")
        return None
    df['published_at'] = pd.to_datetime(df['published_at'])
    outdir = os.path.dirname(path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_parquet(path)
    print(f"\nSuccessfully fetched {len(df)} articles.")
    print(f"News data saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', default=DEFAULT_API_KEY, help='NewsAPI key. Required.')
    parser.add_argument('--output', default='data/news_headlines.parquet', help='Output path for news parquet.')
    parser.add_argument('--max_articles', type=int, default=10, help='Max articles per company [1-100].')
    args = parser.parse_args()
    if not args.apikey or args.apikey == 'f260d5675ae449d5a1e6a30ea49a6410':
        print("f260d5675ae449d5a1e6a30ea49a6410")
        return
    company_list = get_sp500_companies()
    if company_list:
        news_data = fetch_all_news(company_list, apikey=args.apikey, max_articles=args.max_articles)
        save_news_data(news_data, path=args.output)

if __name__ == "__main__":
    main()
