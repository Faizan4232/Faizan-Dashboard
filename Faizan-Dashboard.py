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
# Caching the data loading function for improved performance
@st.cache_data
def load_data():
    """
    Loads and preprocesses the master dataset from a Parquet file.
    Returns the dataframe or None if the file is not found.
    """
    try:
        df = pd.read_parquet('data/master_dataset.parquet')
        df['date'] = pd.to_datetime(df['date'])
        # Ensure numeric columns are treated as such
        for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None

# --- Technical Indicator Functions ---
def calculate_sma(data, window_size):
    """Calculates the Simple Moving Average (SMA)."""
    return data['close'].rolling(window=window_size).mean()

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
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
    
    stock_df = df[df['ticker'] == selected_ticker].copy()

    if stock_df.empty:
        st.warning("No data available for the selected stock.")
        return

    # --- Key Metrics ---
    latest_data = stock_df.sort_values('date').iloc[-1]
    prev_day_data = stock_df.sort_values('date').iloc[-2]
    price_change = latest_data['close'] - prev_day_data['close']
    percent_change = (price_change / prev_day_data['close']) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close Price", f"${latest_data['close']:.2f}", f"{price_change:.2f} ({percent_change:.2f}%)")
    col2.metric("52-Week High", f"${stock_df['high'].max():.2f}")
    col3.metric("52-Week Low", f"${stock_df['low'].min():.2f}")

    # --- Interactive Price Chart with Plotly ---
    st.subheader("Interactive Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_df['date'],
                               open=stock_df['open'],
                               high=stock_df['high'],
                               low=stock_df['low'],
                               close=stock_df['close'],
                               name='Price'))

    # --- Technical Indicators Overlay ---
    st.sidebar.header("Technical Indicators")
    show_sma_20 = st.sidebar.checkbox("Show 20-Day SMA")
    show_sma_50 = st.sidebar.checkbox("Show 50-Day SMA")
    
    if show_sma_20:
        stock_df['SMA_20'] = calculate_sma(stock_df, 20)
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_20'], mode='lines', name='20-Day SMA'))
        
    if show_sma_50:
        stock_df['SMA_50'] = calculate_sma(stock_df, 50)
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['SMA_50'], mode='lines', name='50-Day SMA'))
        
    fig.update_layout(xaxis_rangeslider_visible=False, title=f'{selected_ticker} Price Action')
    st.plotly_chart(fig, use_container_width=True)

    # --- RSI Chart ---
    st.subheader("Relative Strength Index (RSI)")
    stock_df['RSI'] = calculate_rsi(stock_df)
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['RSI'], mode='lines', name='RSI'))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    st.plotly_chart(rsi_fig, use_container_width=True)

def render_prediction_page(df):
    """Renders the AI prediction model page."""
    st.header("🤖 AI Prediction Model")
    
    tickers = sorted(df['ticker'].unique())
    selected_ticker = st.selectbox("Select a Stock to Predict", tickers)
    
    if st.button(f"Run Prediction for {selected_ticker}"):
        with st.spinner('Analyzing market data and running the AI model...'):
            import time
            time.sleep(3) # Simulate model running
            
            # Placeholder for your actual model's prediction
            # In a real scenario, you would call your trained model here.
            # prediction = model.predict(get_latest_data(selected_ticker))
            # confidence = model.predict_proba(...)
            
            prediction = "UP 📈"
            confidence = 0.78
            
            st.success(f"Prediction for {selected_ticker} is complete!")
            st.metric("Predicted Trend for Next Day", prediction, f"Confidence: {confidence:.2%}")
            st.info("Disclaimer: This is a simulated prediction based on a placeholder model. Do not use for actual trading decisions.")

# --- Main App ---
def main():
    """Main function to run the Streamlit app."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Deep Dive Analysis", "AI Prediction Model"])

    master_df = load_data()

    if master_df is None:
        st.error("Fatal Error: The master dataset 'data/master_dataset.parquet' was not found.")
        st.info("Please ensure you have run the data collection and merging scripts first.")
        return

    if page == "Deep Dive Analysis":
        render_deep_dive_page(master_df)
    elif page == "AI Prediction Model":
        render_prediction_page(master_df)

if __name__ == "__main__":
    main()