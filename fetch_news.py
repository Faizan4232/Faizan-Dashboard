import pandas as pd
import os
import requests
from newsapi import NewsApiClient

# --- CONFIGURATION ---
API_KEY = 'f260d5675ae449d5a1e6a30ea49a6410' 

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

def fetch_all_news(companies):
    """Fetches news articles for a list of companies."""
    newsapi = NewsApiClient(api_key=API_KEY)
    all_articles = []
    print("\nFetching news articles... This may take a while.")
    for company in companies:
        ticker = company['Symbol']
        company_name = company['Security']
        try:
            articles = newsapi.get_everything(
                q=f'"{company_name}"',
                language='en',
                sort_by='publishedAt',
                page_size=10 # Limit to 10 recent articles per company for API efficiency
            )
            for article in articles['articles']:
                all_articles.append({
                    'ticker': ticker,
                    'published_at': article['publishedAt'],
                    'title': article['title']
                })
        except Exception as e:
            print(f"Could not fetch news for {company_name}. Error: {e}")
    return pd.DataFrame(all_articles)

def save_news_data(df, path='data/news_headlines.parquet'):
    """Saves the news DataFrame to a Parquet file."""
    if df.empty:
        print("\nNo news articles fetched. Nothing to save.")
        return
    df['published_at'] = pd.to_datetime(df['published_at'])
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_parquet(path)
    print(f"\nSuccessfully fetched {len(df)} articles.")
    print(f"News data saved to {path}")

if __name__ == "__main__":
    company_list = get_sp500_companies()
    if company_list:
        news_data = fetch_all_news(company_list)
        save_news_data(news_data)