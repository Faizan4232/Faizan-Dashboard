import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Market Trend Analysis & Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

@st.cache_data
def generate_mock_data(symbol, period):
    """Generate mock OHLCV stock data for demonstration."""
    np.random.seed(42)

    if period == "1D":
        intervals = 50
    elif period == "1W":
        intervals = 50
    elif period == "1M":
        intervals = 30
    else:
        intervals = 90

    dates = pd.date_range(
        end=datetime.now(),
        periods=intervals,
        freq="D" if intervals > 24 else "H",
    )
    dates = dates[-intervals:]

    price = 150.0
    prices, volumes, opens, highs, lows = [], [], [], [], []

    for i in range(intervals):
        change = np.random.normal(0, 2)
        if i % 5 == 0:
            change += np.random.choice([-3, 3], p=[0.5, 0.5])
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

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        }
    )
    df.set_index("Date", inplace=True)
    return df


def calculate_indicators(df):
    """Calculate a set of technical indicators for visualization."""
    min_length = 20
    if len(df) < min_length:
        st.warning(
            f"Insufficient data ({len(df)} rows) for full indicators. "
            "Showing basic price data only."
        )
        return df

    df = df.copy()

    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = (
        ta.trend.sma_indicator(df["Close"], window=50)
        if len(df) >= 50
        else np.nan
    )
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(
        df["Close"], window=20, window_dev=2
    )
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Lower"] = bollinger.bollinger_lband()
    df["BB_Middle"] = bollinger.bollinger_mavg()

    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["Williams_R"] = ta.momentum.williams_r(
        df["High"], df["Low"], df["Close"], lbp=14
    )

    if len(df) >= 14:
        adx = ta.trend.ADXIndicator(
            df["High"], df["Low"], df["Close"], window=14
        )
        df["ADX"] = adx.adx()
        df["ADX_Pos"] = adx.adx_pos()
        df["ADX_Neg"] = adx.adx_neg()
    else:
        df["ADX"] = df["ADX_Pos"] = df["ADX_Neg"] = np.nan

    if len(df) >= 26:
        ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        df["Ichimoku_A"] = ichimoku.ichimoku_a()
        df["Ichimoku_B"] = ichimoku.ichimoku_b()
        df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
        df["Ichimoku_Conversion"] = ichimoku.ichimoku_conversion_line()
    else:
        df["Ichimoku_A"] = df["Ichimoku_B"] = np.nan
        df["Ichimoku_Base"] = df["Ichimoku_Conversion"] = np.nan

    return df


@st.cache_data
def load_experiment_results():
    """Load precomputed LSTM experiment metrics (with and without sentiment)."""
    df_no = None
    df_with = None

    if os.path.exists("data/results_without_sentiment.parquet"):
        df_no = pd.read_parquet("data/results_without_sentiment.parquet")
    if os.path.exists("data/results_with_sentiment.parquet"):
        df_with = pd.read_parquet("data/results_with_sentiment.parquet")

    return df_no, df_with


@st.cache_data
def load_feature_importance():
    """Load feature importance or selected indicators (if available)."""
    path = "data/feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_daily_sentiment():
    """Load daily sentiment time series for visualization (optional)."""
    path = "data/news_sentiment.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        # Expect columns: ticker, date, sentiment
        return df
    return None


# --------------------------------------------------
# Main app
# --------------------------------------------------

def main():
    st.title("📈 Stock Trend & Sentiment Dashboard")
    st.markdown(
        """
This dashboard visualizes **trend prediction performance** and 
**news-based sentiment** on precomputed experiment results, 
supporting the research pipeline (Spark + indicators + LSTM + sentiment).
"""
    )
    st.info(
        "The dashboard is a visualization and analysis layer. "
        "All model training and feature engineering are performed "
        "in separate research scripts."
    )

    # Sidebar controls (purely for visualization/demo)
    st.sidebar.header("Controls")
    selected_stock = st.sidebar.selectbox(
        "Select Stock (for demo charts)",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        index=0,
    )
    time_frame = st.sidebar.selectbox(
        "Demo Time Frame",
        options=["1D", "1W", "1M", "3M"],
        index=2,
    )
    data_source = st.sidebar.selectbox(
        "Price Data Source (for demo chart only)",
        options=["mock", "real"],
        index=0,
    )

    if st.sidebar.button("Refresh Demo Data"):
        st.cache_data.clear()
        st.rerun()

    # --------------------------------------------------
    # Section 1: LSTM experiment metrics (paper results)
    # --------------------------------------------------
    st.subheader("Trend Prediction Performance (LSTM Experiments)")

    df_no_sent, df_with_sent = load_experiment_results()

    if df_no_sent is None or df_with_sent is None:
        st.warning(
            "Experiment result files not found. "
            "Please run train_lstm.py to generate:\n"
            "- data/results_without_sentiment.parquet\n"
            "- data/results_with_sentiment.parquet"
        )
    else:
        # Expect single-row summary tables or one row per ticker; use first row for global summary
        row_no = df_no_sent.iloc[0]
        row_with = df_with_sent.iloc[0]

        da_no = float(row_no.get("directional_accuracy", np.nan))
        da_with = float(row_with.get("directional_accuracy", np.nan))
        mae_no = float(row_no.get("mae", np.nan))
        mae_with = float(row_with.get("mae", np.nan))
        rmse_no = float(row_no.get("rmse", np.nan))
        rmse_with = float(row_with.get("rmse", np.nan))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("DA (No Sentiment)", f"{da_no:.2%}")
        with col2:
            st.metric("DA (With Sentiment)", f"{da_with:.2%}")
        with col3:
            delta_da = da_with - da_no
            st.metric("Δ Directional Accuracy", f"{delta_da:.2%}")

        st.markdown("**Error Metrics (Test Set)**")
        col4, col5 = st.columns(2)
        with col4:
            st.write("Without Sentiment")
            st.write(f"MAE:  {mae_no:.4f}")
            st.write(f"RMSE: {rmse_no:.4f}")
        with col5:
            st.write("With Sentiment")
            st.write(f"MAE:  {mae_with:.4f}")
            st.write(f"RMSE: {rmse_with:.4f}")

    st.markdown("---")

    # --------------------------------------------------
    # Section 2: Feature importance / selected indicators
    # --------------------------------------------------
    st.subheader("Selected Technical Indicators / Feature Importance")

    fi = load_feature_importance()
    if fi is None:
        st.info(
            "Feature importance file not found. "
            "Once feature_engineering.py saves 'data/feature_importance.csv', "
            "it will be displayed here."
        )
    else:
        st.dataframe(
            fi.sort_values(
                fi.columns[-1], ascending=False
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # --------------------------------------------------
    # Section 3: Price + indicators demo chart (for one stock)
    # --------------------------------------------------
    st.subheader("Price & Technical Indicators (Demo View)")

    # Load demo price data (not used for paper metrics)
    with st.spinner("Loading demo price data..."):
        if data_source == "real":
            ticker = yf.Ticker(selected_stock)
            if time_frame == "1D":
                demo_df = ticker.history(period="5d", interval="1h")
            elif time_frame == "1W":
                demo_df = ticker.history(period="1mo", interval="1d")
            else:
                demo_df = ticker.history(period=time_frame.lower())

            if demo_df.empty or len(demo_df) < 5:
                st.warning(
                    "Real data fetch returned insufficient rows. "
                    "Falling back to mock data."
                )
                demo_df = generate_mock_data(selected_stock, time_frame)
        else:
            demo_df = generate_mock_data(selected_stock, time_frame)

    demo_df = calculate_indicators(demo_df)

    if not demo_df.empty:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
        )

        # Candlestick with SMA / Bollinger
        fig.add_trace(
            go.Candlestick(
                x=demo_df.index,
                open=demo_df["Open"],
                high=demo_df["High"],
                low=demo_df["Low"],
                close=demo_df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
        if "SMA_20" in demo_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["SMA_20"],
                    mode="lines",
                    name="SMA 20",
                    line=dict(color="orange"),
                ),
                row=1,
                col=1,
            )
        if "BB_Upper" in demo_df.columns and "BB_Lower" in demo_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["BB_Upper"],
                    mode="lines",
                    name="BB Upper",
                    line=dict(color="gray", width=1),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["BB_Lower"],
                    mode="lines",
                    name="BB Lower",
                    line=dict(color="gray", width=1),
                ),
                row=1,
                col=1,
            )

        # RSI
        if "RSI" in demo_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["RSI"],
                    mode="lines",
                    name="RSI",
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )

        # MACD
        if "MACD" in demo_df.columns and "MACD_Signal" in demo_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["MACD"],
                    mode="lines",
                    name="MACD",
                    line=dict(color="blue"),
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=demo_df.index,
                    y=demo_df["MACD_Signal"],
                    mode="lines",
                    name="MACD Signal",
                    line=dict(color="red"),
                ),
                row=3,
                col=1,
            )

        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------
    # Section 4: Sentiment timeline (optional visualization)
    # --------------------------------------------------
    st.subheader("Daily News Sentiment (Aggregated)")

    sentiment_df = load_daily_sentiment()
    if sentiment_df is None:
        st.info(
            "Daily sentiment file not found. "
            "Once fetch_news.py saves 'data/news_sentiment.parquet', "
            "aggregated sentiment will be shown here."
        )
    else:
        # If multiple tickers exist, allow filter
        tickers = sorted(sentiment_df["ticker"].unique())
        selected_ticker = st.selectbox(
            "Select Ticker for Sentiment View", options=tickers
        )
        sent_sub = sentiment_df[sentiment_df["ticker"] == selected_ticker]

        sent_sub = sent_sub.sort_values("date")
        fig_sent = go.Figure()
        fig_sent.add_trace(
            go.Scatter(
                x=sent_sub["date"],
                y=sent_sub["sentiment"],
                mode="lines+markers",
                name="Daily Sentiment",
            )
        )
        fig_sent.update_layout(
            height=300,
            template="plotly_white",
            yaxis_title="Sentiment (polarity)",
        )
        st.plotly_chart(fig_sent, use_container_width=True)


if __name__ == "__main__":
    main()
