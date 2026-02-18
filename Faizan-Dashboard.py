import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import os
from tensorflow.keras.models import load_model

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Stock Market Trend Analysis & Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# LOAD OFFLINE DATA
# ==================================================
@st.cache_data
def load_price_data():
    path = "data/master_dataset.parquet"
    if not os.path.exists(path):
        st.error("❌ master_dataset.parquet not found. Run feature_engineering.py")
        st.stop()

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_feature_importance():
    path = "data/feature_importance.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_sentiment():
    path = "data/news_sentiment.parquet"
    return pd.read_parquet(path) if os.path.exists(path) else None


# ==================================================
# TECHNICAL INDICATORS
# ==================================================
def calculate_indicators(df):
    df = df.copy()

    if len(df) < 20:
        return df

    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    if len(df) >= 26:
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

    return df


# ==================================================
# MAIN APP
# ==================================================
def main():

    df_all = load_price_data()

    # -------------------------------
    # SIDEBAR (OLD-STYLE CONTROLS)
    # -------------------------------
    st.sidebar.title("Controls")

    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        sorted(df_all["Company"].unique())
    )

    time_frame = st.sidebar.selectbox(
        "Time Frame",
        ["1W", "1M", "3M", "6M", "1Y", "ALL"]
    )

    chart_mode = st.sidebar.selectbox(
        "Chart Mode",
        ["Candlestick", "Line"]
    )

    if st.sidebar.button("Refresh View"):
        st.cache_data.clear()
        st.rerun()

    # -------------------------------
    # FILTER DATA (OFFLINE)
    # -------------------------------
    stock_df = df_all[df_all["Company"] == selected_stock].sort_values("Date")

    latest_date = stock_df["Date"].max()

    if time_frame == "1W":
        stock_df = stock_df[stock_df["Date"] >= latest_date - pd.Timedelta(days=7)]
    elif time_frame == "1M":
        stock_df = stock_df[stock_df["Date"] >= latest_date - pd.Timedelta(days=30)]
    elif time_frame == "3M":
        stock_df = stock_df[stock_df["Date"] >= latest_date - pd.Timedelta(days=90)]
    elif time_frame == "6M":
        stock_df = stock_df[stock_df["Date"] >= latest_date - pd.Timedelta(days=180)]
    elif time_frame == "1Y":
        stock_df = stock_df[stock_df["Date"] >= latest_date - pd.Timedelta(days=365)]

    stock_df = calculate_indicators(stock_df)
    stock_df = stock_df.set_index("Date")

    # -------------------------------
    # HEADER
    # -------------------------------
    st.title("📈 Stock Market Trend Analysis & Sentiment Dashboard")

    st.info(
        "This dashboard runs fully offline using locally stored CSV/Parquet files. "
        "All model training and feature engineering are executed in separate scripts."
    )

    # -------------------------------
    # PRICE + INDICATORS
    # -------------------------------
    st.subheader("Price & Technical Indicators")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25]
    )

    if chart_mode == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=stock_df.index,
                open=stock_df["Open"],
                high=stock_df["High"],
                low=stock_df["Low"],
                close=stock_df["Close"],
                name="Price"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index,
                y=stock_df["Close"],
                mode="lines",
                name="Close Price"
            ),
            row=1, col=1
        )

    if "SMA_20" in stock_df:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index,
                y=stock_df["SMA_20"],
                name="SMA 20"
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=stock_df.index,
            y=stock_df["RSI"],
            name="RSI"
        ),
        row=2, col=1
    )

    if "MACD" in stock_df:
        fig.add_trace(
            go.Scatter(
                x=stock_df.index,
                y=stock_df["MACD"],
                name="MACD"
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=stock_df.index,
                y=stock_df["MACD_Signal"],
                name="MACD Signal"
            ),
            row=3, col=1
        )

    fig.update_layout(
        height=800,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------
    st.subheader("Selected Technical Indicators / Feature Importance")

    fi = load_feature_importance()
    if fi is not None:
        st.dataframe(fi, use_container_width=True)
    else:
        st.info("Feature importance file not found.")

    # -------------------------------
    # SENTIMENT
    # -------------------------------
    st.subheader("Daily News Sentiment")

    sent_df = load_sentiment()
    if sent_df is not None:
        ticker = st.selectbox(
            "Select Ticker",
            sorted(sent_df["ticker"].unique())
        )

        sub = sent_df[sent_df["ticker"] == ticker].sort_values("date")

        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["sentiment"],
                mode="lines+markers",
                name="Sentiment"
            )
        )
        fig_s.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_s, use_container_width=True)

        st.caption(
            "Note: Sentiment values are aggregated daily and may appear neutral "
            "for periods with limited or balanced news coverage."
        )
    else:
        st.info("Sentiment data not available.")


# ==================================================
if __name__ == "__main__":
    main()
