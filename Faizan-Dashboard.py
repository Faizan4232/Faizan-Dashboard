import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Stock Trend Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- Data Caching ---
@st.cache_data
def load_data():
    """Loads and preprocesses the master dataset."""
    try:
        df = pd.read_parquet('data/master_dataset.parquet')
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        return df
    except FileNotFoundError:
        return None

# --- Technical Indicator Functions ---
def calculate_sma(data, window_size):
    return data['close'].rolling(window=window_size).mean()

def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Page Rendering Functions ---
def render_deep_dive_page(df):
    """Renders the detailed analysis page for a single stock."""
    st.header("📊 Deep Dive Stock Analysis")
    
    tickers = sorted(df['ticker'].unique())
    selected_ticker = st.selectbox("Select a Stock Ticker", tickers)
    
    stock_df = df[df['ticker'] == selected_ticker].copy().sort_values('date')

    if stock_df.empty or len(stock_df) < 2:
        st.warning("Not enough data available for the selected stock.")
        return

    # Key Metrics
    latest_data = stock_df.iloc[-1]
    prev_day_data = stock_df.iloc[-2]
    price_change = latest_data['close'] - prev_day_data['close']
    percent_change = (price_change / prev_day_data['close']) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close Price", f"${latest_data['close']:.2f}", f"{price_change:.2f} ({percent_change:.2f}%)")
    col2.metric("52-Week High", f"${stock_df['close'].max():.2f}")
    col3.metric("52-Week Low", f"${stock_df['close'].min():.2f}")

    # Interactive Price Chart
    st.subheader("Interactive Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Close Price'))

    # Technical Indicators
    st.sidebar.header("Technical Indicators")
    if st.sidebar.checkbox("Show 20-Day SMA"):
        stock_df['SMA_20'] = calculate_sma(stock_df, 20)
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_20'], mode='lines', name='20-Day SMA'))
    if st.sidebar.checkbox("Show 50-Day SMA"):
        stock_df['SMA_50'] = calculate_sma(stock_df, 50)
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_50'], mode='lines', name='50-Day SMA'))
        
    fig.update_layout(title=f'{selected_ticker} Price Action', yaxis_title='Price (USD)')
    st.plotly_chart(fig, use_container_width=True)

def render_prediction_page(df):
    """Renders the AI prediction model page."""
    st.header("🤖 AI Prediction Model")
    
    tickers = sorted(df['ticker'].unique())
    selected_ticker = st.selectbox("Select a Stock to Predict", tickers)
    
    if st.button(f"Run Prediction for {selected_ticker}"):
        with st.spinner('Analyzing...'):
            import time
            time.sleep(2)
            prediction = "UP 📈"
            confidence = 0.78
            st.success(f"Prediction for {selected_ticker}: **{prediction}** with {confidence:.2%} confidence.")
            st.info("Disclaimer: This is a simulated prediction.")

# --- Main App ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Deep Dive Analysis", "AI Prediction Model"])
    master_df = load_data()

    if master_df is None:
        st.error("Fatal Error: 'master_dataset.parquet' not found.")
        st.info("Please run the data scripts first: `download_data.py`, `fetch_news.py`, then `merge_data.py`.")
        return

    if page == "Deep Dive Analysis":
        render_deep_dive_page(master_df)
    elif page == "AI Prediction Model":
        render_prediction_page(master_df)

if __name__ == "__main__":
    main()