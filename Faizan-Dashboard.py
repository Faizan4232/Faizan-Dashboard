# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import yfinance as yf
# from datetime import datetime, timedelta
# import ta
# import os

# # ==================================================
# # PAGE CONFIG
# # ==================================================
# st.set_page_config(
#     page_title="Stock Market Trend Analysis & Sentiment Dashboard",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ==================================================
# # MOCK DATA GENERATION (FOR DEMO)
# # ==================================================
# @st.cache_data
# def generate_mock_data(symbol, period):
#     np.random.seed(42)

#     if period == "1D":
#         intervals = 50
#     elif period == "1W":
#         intervals = 50
#     elif period == "1M":
#         intervals = 30
#     else:
#         intervals = 90

#     dates = pd.date_range(
#         end=datetime.now(),
#         periods=intervals,
#         freq="D"
#     )

#     price = 150.0
#     opens, highs, lows, closes, volumes = [], [], [], [], []

#     for i in range(intervals):
#         change = np.random.normal(0, 2)
#         price += change

#         o = price + np.random.normal(0, 1)
#         h = max(o, price) + abs(np.random.normal(0, 1.5))
#         l = min(o, price) - abs(np.random.normal(0, 1.5))
#         v = np.random.randint(5_000_000, 20_000_000)

#         opens.append(o)
#         highs.append(h)
#         lows.append(l)
#         closes.append(price)
#         volumes.append(v)

#     df = pd.DataFrame(
#         {
#             "Open": opens,
#             "High": highs,
#             "Low": lows,
#             "Close": closes,
#             "Volume": volumes,
#         },
#         index=dates
#     )

#     return df


# # ==================================================
# # TECHNICAL INDICATORS (SAFE VERSION)
# # ==================================================
# def calculate_indicators(df):
#     df = df.copy()

#     if len(df) < 20:
#         st.warning(
#             f"Only {len(df)} rows available. "
#             "Indicators disabled for short windows."
#         )
#         return df

#     # -----------------------------
#     # Moving Averages
#     # -----------------------------
#     df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
#     df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50) if len(df) >= 50 else np.nan

#     # -----------------------------
#     # RSI
#     # -----------------------------
#     df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

#     # -----------------------------
#     # MACD
#     # -----------------------------
#     if len(df) >= 26:
#         macd = ta.trend.MACD(df["Close"])
#         df["MACD"] = macd.macd()
#         df["MACD_Signal"] = macd.macd_signal()
#     else:
#         df["MACD"] = df["MACD_Signal"] = np.nan

#     # -----------------------------
#     # Bollinger Bands
#     # -----------------------------
#     bb = ta.volatility.BollingerBands(df["Close"], window=20)
#     df["BB_Upper"] = bb.bollinger_hband()
#     df["BB_Lower"] = bb.bollinger_lband()
#     df["BB_Middle"] = bb.bollinger_mavg()

#     # -----------------------------
#     # Stochastic Oscillator
#     # -----------------------------
#     if len(df) >= 17:
#         stoch = ta.momentum.StochasticOscillator(
#             df["High"], df["Low"], df["Close"], window=14, smooth_window=3
#         )
#         df["Stoch_K"] = stoch.stoch()
#         df["Stoch_D"] = stoch.stoch_signal()
#     else:
#         df["Stoch_K"] = df["Stoch_D"] = np.nan

#     # -----------------------------
#     # Williams %R
#     # -----------------------------
#     df["Williams_R"] = (
#         ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=14)
#         if len(df) >= 14 else np.nan
#     )

#     # -----------------------------
#     # ADX (FIXED)
#     # -----------------------------
#     if len(df) >= 30:
#         adx = ta.trend.ADXIndicator(
#             high=df["High"],
#             low=df["Low"],
#             close=df["Close"],
#             window=14
#         )
#         df["ADX"] = adx.adx()
#         df["ADX_Pos"] = adx.adx_pos()
#         df["ADX_Neg"] = adx.adx_neg()
#     else:
#         df["ADX"] = df["ADX_Pos"] = df["ADX_Neg"] = np.nan

#     # -----------------------------
#     # Ichimoku Cloud
#     # -----------------------------
#     if len(df) >= 52:
#         ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"])
#         df["Ichimoku_A"] = ichi.ichimoku_a()
#         df["Ichimoku_B"] = ichi.ichimoku_b()
#         df["Ichimoku_Base"] = ichi.ichimoku_base_line()
#         df["Ichimoku_Conversion"] = ichi.ichimoku_conversion_line()
#     else:
#         df["Ichimoku_A"] = df["Ichimoku_B"] = np.nan
#         df["Ichimoku_Base"] = df["Ichimoku_Conversion"] = np.nan

#     return df


# # ==================================================
# # LOAD PRECOMPUTED RESULTS
# # ==================================================
# @st.cache_data
# def load_results(path):
#     return pd.read_parquet(path) if os.path.exists(path) else None


# @st.cache_data
# def load_feature_importance():
#     path = "data/feature_importance.csv"
#     return pd.read_csv(path) if os.path.exists(path) else None


# @st.cache_data
# def load_sentiment():
#     path = "data/news_sentiment.parquet"
#     return pd.read_parquet(path) if os.path.exists(path) else None


# # ==================================================
# # MAIN APP
# # ==================================================
# def main():

#     st.title("📈 Stock Market Trend Analysis & Sentiment Dashboard")

#     st.markdown(
#         """
# This dashboard visualizes **trend prediction performance** and 
# **news-based sentiment** using precomputed experiment results,
# supporting the research pipeline (Spark + Indicators + LSTM + Sentiment).
# """
#     )

#     st.info(
#         "This dashboard is a visualization layer only. "
#         "All model training and feature engineering are performed "
#         "in separate research scripts."
#     )

#     # --------------------------------------------------
#     # SIDEBAR CONTROLS
#     # --------------------------------------------------
#     st.sidebar.header("Controls")

#     selected_stock = st.sidebar.selectbox(
#         "Select Stock (for demo charts)",
#         ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
#     )

#     time_frame = st.sidebar.selectbox(
#         "Demo Time Frame",
#         ["1D", "1W", "1M", "3M"]
#     )

#     data_source = st.sidebar.selectbox(
#         "Price Data Source (for demo chart only)",
#         ["mock", "real"]
#     )

#     if st.sidebar.button("Refresh Demo Data"):
#         st.cache_data.clear()
#         st.rerun()

#     # --------------------------------------------------
#     # LSTM RESULTS (DA + MAE + RMSE)
#     # --------------------------------------------------
#     st.subheader("Trend Prediction Performance (LSTM Experiments)")

#     df_no = load_results("data/results_without_sentiment.parquet")
#     df_with = load_results("data/results_with_sentiment.parquet")

#     if df_no is not None and df_with is not None:

#         r_no = df_no.iloc[0]
#         r_with = df_with.iloc[0]

#         col1, col2, col3 = st.columns(3)
#         col1.metric("DA (No Sentiment)", f"{r_no['directional_accuracy']:.2%}")
#         col2.metric("DA (With Sentiment)", f"{r_with['directional_accuracy']:.2%}")
#         col3.metric(
#             "Δ Directional Accuracy",
#             f"{(r_with['directional_accuracy'] - r_no['directional_accuracy']):.2%}"
#         )

#         st.markdown("### Error Metrics (Test Set)")
#         col4, col5 = st.columns(2)

#         with col4:
#             st.markdown("**Without Sentiment**")
#             st.write(f"MAE:  {r_no['mae']:.4f}")
#             st.write(f"RMSE: {r_no['rmse']:.4f}")

#         with col5:
#             st.markdown("**With Sentiment**")
#             st.write(f"MAE:  {r_with['mae']:.4f}")
#             st.write(f"RMSE: {r_with['rmse']:.4f}")

#     else:
#         st.warning("Run train_lstm.py to generate experiment results.")

#     st.markdown("---")

#     # --------------------------------------------------
#     # FEATURE IMPORTANCE
#     # --------------------------------------------------
#     st.subheader("Selected Technical Indicators / Feature Importance")

#     fi = load_feature_importance()
#     if fi is not None:
#         st.dataframe(fi, use_container_width=True)
#     else:
#         st.info("Feature importance file not found.")

#     st.markdown("---")

#     # --------------------------------------------------
#     # PRICE + INDICATORS
#     # --------------------------------------------------
#     st.subheader("Price & Technical Indicators (Demo View)")

#     with st.spinner("Loading price data..."):
#         if data_source == "real":
#             demo_df = yf.Ticker(selected_stock).history(period="3mo")
#             if demo_df.empty:
#                 demo_df = generate_mock_data(selected_stock, time_frame)
#         else:
#             demo_df = generate_mock_data(selected_stock, time_frame)

#     demo_df = calculate_indicators(demo_df)

#     fig = make_subplots(
#         rows=3,
#         cols=1,
#         shared_xaxes=True,
#         row_heights=[0.5, 0.25, 0.25]
#     )

#     fig.add_trace(
#         go.Candlestick(
#             x=demo_df.index,
#             open=demo_df["Open"],
#             high=demo_df["High"],
#             low=demo_df["Low"],
#             close=demo_df["Close"],
#             name="Price"
#         ),
#         row=1,
#         col=1
#     )

#     if "SMA_20" in demo_df:
#         fig.add_trace(go.Scatter(
#             x=demo_df.index, y=demo_df["SMA_20"], name="SMA 20"
#         ), row=1, col=1)

#     fig.add_trace(go.Scatter(
#         x=demo_df.index, y=demo_df["RSI"], name="RSI"
#     ), row=2, col=1)

#     fig.add_trace(go.Scatter(
#         x=demo_df.index, y=demo_df["MACD"], name="MACD"
#     ), row=3, col=1)

#     fig.add_trace(go.Scatter(
#         x=demo_df.index, y=demo_df["MACD_Signal"], name="MACD Signal"
#     ), row=3, col=1)

#     fig.update_layout(
#         height=800,
#         template="plotly_white",
#         xaxis_rangeslider_visible=False
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("---")

#     # --------------------------------------------------
#     # SENTIMENT VISUALIZATION
#     # --------------------------------------------------
#     st.subheader("Daily News Sentiment")

#     sent_df = load_sentiment()
#     if sent_df is not None:
#         tickers = sorted(sent_df["ticker"].unique())
#         t = st.selectbox("Select Ticker", tickers)
#         sub = sent_df[sent_df["ticker"] == t].sort_values("date")

#         fig_s = go.Figure()
#         fig_s.add_trace(go.Scatter(
#             x=sub["date"],
#             y=sub["sentiment"],
#             mode="lines+markers",
#             name="Sentiment"
#         ))
#         fig_s.update_layout(height=300, template="plotly_white")
#         st.plotly_chart(fig_s, use_container_width=True)
#     else:
#         st.info("Sentiment data not available.")


# # ==================================================
# if __name__ == "__main__":
#     main()
# ================================
# Faizan-Dashboard.py
# OFFLINE Stock Market Dashboard
# ================================
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
