import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(
    page_title="Stock Market Trend Analysis & Prediction",
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
    """Generate mock stock data for demonstration."""
    np.random.seed(42)
    if period == '1D':
        intervals = 50  # Increased for indicator compatibility
    elif period == '1W':
        intervals = 50  # Increased for indicator compatibility
    elif period == '1M':
        intervals = 30
    else:
        intervals = 90
    
    dates = pd.date_range(end=datetime.now(), periods=intervals, freq='D' if intervals > 24 else 'H')
    dates = dates[-intervals:]
    
    price = 150.0
    prices = []
    volumes = []
    opens = []
    highs = []
    lows = []
    
    for i in range(intervals):
        change = np.random.normal(0, 2)
        if i % 5 == 0:
            change += np.random.choice([-3, 3], p=[0.5, 0.5])  # Fixed: probabilities sum to 1
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
    """Calculate comprehensive technical indicators."""
    min_length = 20  # Minimum rows needed for most indicators
    if len(df) < min_length:
        st.warning(f"Insufficient data ({len(df)} rows) for indicators. Using basic data only. Switch to a longer time frame or real data for full analysis.")
        return df  # Return df without indicators to avoid errors
    
    # Existing: SMA, RSI, MACD
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50) if len(df) >= 50 else np.nan
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # New: Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    # New: Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # New: Williams %R
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    
    # New: ADX (Average Directional Index) - Only if enough data
    if len(df) >= 14:
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()
    else:
        df['ADX'] = df['ADX_Pos'] = df['ADX_Neg'] = np.nan
    
    # New: Ichimoku Cloud (simplified) - Only if enough data
    if len(df) >= 26:  # Ichimoku needs more data
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
    else:
        df['Ichimoku_A'] = df['Ichimoku_B'] = df['Ichimoku_Base'] = df['Ichimoku_Conversion'] = np.nan
    
    return df

def predict_next_price(df, days_ahead=1):
    """Simple linear regression prediction for next price."""
    df = df.dropna()  # Drop NaNs first
    if len(df) < 2:  # Need at least 2 points for regression
        return None, None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_X = np.array([[len(df) + days_ahead - 1]])
    prediction = model.predict(next_X)[0]
    return prediction, model.score(X, y)

def main():
    st.title("📈 Advanced Stock Market Trend Analysis & Prediction Dashboard")
    st.markdown("Interactive dashboard with comprehensive indicators and basic trend prediction.")

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
            ticker = yf.Ticker(selected_stock)
            if time_frame == '1D':
                df = ticker.history(period='5d', interval='1h')  # Fetch more data for 1D to ensure sufficiency
            elif time_frame == '1W':
                df = ticker.history(period='1mo', interval='1d')  # Ensure at least a week's worth
            else:
                df = ticker.history(period=time_frame.lower())
            if df.empty or len(df) < 5:  # Fallback if still too short
                st.warning("Real data fetch returned insufficient data. Using mock data as fallback.")
                df = generate_mock_data(selected_stock, time_frame)
        else:
            df = generate_mock_data(selected_stock, time_frame)
        
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

    # Key Metrics Row (expanded) - With column checks to prevent KeyError
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi = latest['RSI']
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            color_class = "bearish" if rsi > 70 else "bullish" if rsi < 30 else "neutral"
        else:
            rsi = "N/A"
            rsi_status = "Insufficient Data"
            color_class = "neutral"
        st.metric(label="RSI (14)", value=f"{rsi}", delta_color="normal")
        st.caption(f"Status: <span class='{color_class}'>{rsi_status}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'ADX' in df.columns and not pd.isna(latest['ADX']):
            adx = latest['ADX']
            trend_strength = "Strong" if adx > 25 else "Weak"
            color_class = "bullish" if adx > 25 else "neutral"
        else:
            adx = "N/A"
            trend_strength = "Insufficient Data"
            color_class = "neutral"
        st.metric(label="ADX (14)", value=f"{adx}", delta_color="normal")
        st.caption(f"Trend: <span class='{color_class}'>{trend_strength}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Stoch_K' in df.columns and not pd.isna(latest['Stoch_K']):
            stoch_k = latest['Stoch_K']
            stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
            color_class = "bearish" if stoch_k > 80 else "bullish" if stoch_k < 20 else "neutral"
        else:
            stoch_k = "N/A"
            stoch_status = "Insufficient Data"
            color_class = "neutral"
        st.metric(label="Stoch %K", value=f"{stoch_k}", delta_color="normal")
        st.caption(f"Status: <span class='{color_class}'>{stoch_status}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        prediction, r2 = predict_next_price(df)
        if prediction:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Predicted Next Price", value=f"${prediction:.2f}", delta_color="normal")
            st.caption(f"Model R²: {r2:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Charts (expanded) - Handle missing indicators gracefully
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Price Chart with Indicators - {selected_stock}")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="OHLC"
        ))
        if 'SMA_20' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='orange')))
        if 'BB_Upper' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='green', dash='dash')))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='red', dash='dash')))
        if 'Ichimoku_A' in df.columns:
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_A'], name="Ichimoku A", line=dict(color='purple')))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_B'], name="Ichimoku B", line=dict(color='blue')))
        fig_price.update_layout(
            title=f"{selected_stock} Price with Indicators",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Chart")
        if prediction:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical Close", mode='lines'))
            future_date = df.index[-1] + pd.Timedelta(days=1)
            fig_pred.add_trace(go.Scatter(x=[future_date], y=[prediction], name="Predicted", mode='markers', marker=dict(color='red', size=10)))
            fig_pred.update_layout(
                title="Price Prediction (Linear Regression)",
                xaxis_title="Date", yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.write("Not enough data for prediction.")

    # Technical Indicators Subplot (expanded) - Handle missing indicators
    st.subheader("Comprehensive Technical Indicators")
    fig_indicators = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price & MAs', 'RSI & Williams %R', 'MACD & ADX', 'Stoch & Ichimoku'),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )
    
    # Price
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='blue')), row=1, col=1)
    if 'SMA_20' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='orange')), row=1, col=1)
    
    # RSI & Williams
    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    if 'RSI' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    if 'Williams_R' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Williams_R'], name="Williams %R", line=dict(color='brown')), row=2, col=1)
    
    # MACD & ADX
    if 'MACD' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')), row=3, col=1)
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='red')), row=3, col=1)
    if 'ADX' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX", line=dict(color='green')), row=3, col=1)
    
    # Stoch & Ichimoku
    if 'Stoch_K' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name="Stoch %K", line=dict(color='orange')), row=4, col=1)
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name="Stoch %D", line=dict(color='yellow')), row=4, col=1)
    if 'Ichimoku_Conversion' in df.columns:
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Conversion'], name="Ichimoku Conversion", line=dict(color='purple')), row=4, col=1)
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Base'], name="Ichimoku Base", line=dict(color='blue')), row=4, col=1)
    
    fig_indicators.update_layout(height=800, showlegend=True)
