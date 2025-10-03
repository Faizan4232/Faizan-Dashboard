import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf  # For real data (optional; we'll use mock for demo)
from datetime import datetime, timedelta
import ta  # Technical Analysis library (pip install ta)

# Page config
st.set_page_config(
    page_title="Stock Market Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bullish { color: #28a745; }
    .bearish { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_mock_data(symbol, period):
    """Generate mock stock data for demonstration. Replace with real API calls."""
    np.random.seed(42)  # For reproducibility
    if period == '1D':
        days = 1
        intervals = 24  # Hourly
    elif period == '1W':
        days = 7
        intervals = 35  # ~5 per day
    elif period == '1M':
        days = 30
        intervals = 30  # Daily
    else:  # 3M
        days = 90
        intervals = 90  # Daily
    
    dates = pd.date_range(end=datetime.now(), periods=intervals, freq='D' if intervals > 24 else 'H')
    dates = dates[-intervals:]  # Ensure correct length
    
    # Generate price data with some trend and noise
    price = 150.0
    prices = []
    volumes = []
    opens = []
    highs = []
    lows = []
    
    for i in range(intervals):
        change = np.random.normal(0, 2)  # Random walk with drift
        if i % 5 == 0:  # Occasional trend
            change += np.random.choice([-3, 3], p=[0.3, 0.3])
        price += change
        open_price = price + np.random.normal(0, 1)
        high = max(open_price, price) + abs(np.random.normal(0, 1.5))
        low = min(open_price, price) - abs(np.random.normal(0, 1.5))
        volume = np.random.randint(5_000_000, 20_000_000)
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        prices.append(price)
        volumes.append(volume)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    df.set_index('Date', inplace=True)
    return df

def calculate_indicators(df):
    """Calculate technical indicators using TA library."""
    # Simple Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    return df

def main():
    st.title("📈 Advanced Stock Market Trend Analysis Dashboard")
    st.markdown("Interactive dashboard for stock price analysis, technical indicators, and market trends.")

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'],
        index=0
    )
    time_frame = st.sidebar.selectbox(
        "Time Frame",
        options=['1D', '1W', '1M', '3M'],
        index=2
    )
    use_real_data = st.sidebar.checkbox("Use Real Data (via yfinance)", value=False)
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Loading market data..."):
        if use_real_data:
            # Real data fetch (requires internet)
            ticker = yf.Ticker(selected_stock)
            if time_frame == '1D':
                df = ticker.history(period='1d', interval='1h')
            elif time_frame == '1W':
                df = ticker.history(period='5d', interval='1h')  # Approx 1 week
            else:
                df = ticker.history(period=time_frame.lower())
        else:
            df = generate_mock_data(selected_stock, time_frame)
        
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Current Price",
            value=f"${latest['Close']:.2f}",
            delta=f"${latest['Close'] - prev['Close']:.2f} ({((latest['Close'] - prev['Close']) / prev['Close'] * 100):+.2f}%)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        sma20 = latest['SMA_20']
        trend20 = "bullish" if latest['Close'] > sma20 else "bearish"
        st.metric(
            label="SMA (20)",
            value=f"${sma20:.2f}",
            delta_color="normal"
        )
        st.caption(f"Trend: <span class='{trend20}'>{trend20.upper()}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        rsi = latest['RSI']
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        color_class = "bearish" if rsi > 70 else "bullish" if rsi < 30 else "neutral"
        st.metric(label="RSI (14)", value=f"{rsi:.2f}", delta_color="normal")
        st.caption(f"Status: <span class='{color_class}'>{rsi_status}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        macd_val = latest['MACD']
        macd_trend = "Bullish" if macd_val > latest['MACD_Signal'] else "Bearish"
        color_class = "bullish" if macd_trend == "Bullish" else "bearish"
        st.metric(label="MACD", value=f"{macd_val:.4f}", delta_color="normal")
        st.caption(f"Signal: <span class='{color_class}'>{macd_trend}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Price Chart - {selected_stock}")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="OHLC"
        ))
        fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='orange')))
        if len(df) >= 50:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='red')))
        fig_price.update_layout(
            title=f"{selected_stock} Price and Moving Averages",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Volume Analysis")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='lightblue'))
        fig_volume.update_layout(
            title=f"{selected_stock} Trading Volume",
            xaxis_title="Date", yaxis_title="Volume",
            height=400
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    # Technical Indicators Subplot
    st.subheader("Technical Indicators")
    fig_indicators = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & MAs', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='blue')), row=1, col=1)
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='orange')), row=1, col=1)
    
    # RSI
    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    
    # MACD
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red')), row=3, col=1)
    
    fig_indicators.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig_indicators, use_container_width=True)

    # Data Table
    st.subheader("Recent Market Data")
    df_display = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].tail(10).round(2)
    st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()
