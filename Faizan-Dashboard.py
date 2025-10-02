# app.py
"""
Streamlit Stock Prediction Dashboard
- Fetches historical data with yfinance
- Computes indicators + lag features
- Trains RandomForest or ARIMA baseline
- Interactive Plotly charts and export buttons
"""
import streamlit as st
st.set_page_config(layout="wide", page_title="Stock Predictor — Streamlit", page_icon="📈")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import RandomizedSearchCV
import joblib
import io

# ---------------------------
# Helpers / Feature Engineering
# ---------------------------
@st.cache_data(ttl=3600)
def download_data(ticker: str, period: str = "5y", interval: str = "1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=lambda s: s.strip())
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common indicators (SMA, EMA, returns, volatility, MACD, RSI)"""
    out = df.copy()
    out['adj_close'] = out['Adj Close']
    out['return_1d'] = out['adj_close'].pct_change()
    out['log_return_1d'] = np.log1p(out['return_1d'])
    # SMA
    out['sma_7'] = out['adj_close'].rolling(7).mean()
    out['sma_21'] = out['adj_close'].rolling(21).mean()
    # EMA
    out['ema_12'] = out['adj_close'].ewm(span=12, adjust=False).mean()
    out['ema_26'] = out['adj_close'].ewm(span=26, adjust=False).mean()
    # MACD
    out['macd'] = out['ema_12'] - out['ema_26']
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    # Volatility
    out['volatility_21'] = out['return_1d'].rolling(21).std()
    # RSI (14)
    delta = out['adj_close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    out['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    # Volume-based features
    out['vol_change'] = out['Volume'].pct_change()
    out['vol_21'] = out['Volume'].rolling(21).mean()
    return out

def create_lag_features(df: pd.DataFrame, lags=10):
    df2 = df.copy()
    for i in range(1, lags+1):
        df2[f'lag_{i}'] = df2['adj_close'].shift(i)
    # rolling features for last windows
    df2['rolling_max_21'] = df2['adj_close'].rolling(21).max().shift(1)
    df2['rolling_min_21'] = df2['adj_close'].rolling(21).min().shift(1)
    df2['rolling_mean_7'] = df2['adj_close'].rolling(7).mean().shift(1)
    df2['day_of_week'] = df2.index.dayofweek
    return df2

def prepare_ml_data(df: pd.DataFrame, horizon=1, lags=10):
    """Create features and target for supervised learning.
       horizon=1 -> predict next day's adj_close
    """
    df = add_technical_indicators(df)
    df = create_lag_features(df, lags=lags)
    df['target'] = df['adj_close'].shift(-horizon)  # next-day (or horizon) price
    df = df.dropna().copy()
    feature_cols = [c for c in df.columns if c.startswith('lag_') or c in [
        'sma_7','sma_21','ema_12','ema_26','macd','macd_signal','volatility_21','rsi_14','vol_change','vol_21','rolling_max_21','rolling_min_21','rolling_mean_7','day_of_week']]
    X = df[feature_cols]
    y = df['target']
    return X, y, df

# ---------------------------
# UI: Sidebar controls
# ---------------------------
st.sidebar.header("Data & Model controls")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="AAPL").upper()
period = st.sidebar.selectbox("History period", options=["1y","2y","3y","5y","10y","max"], index=3)
interval = st.sidebar.selectbox("Interval", options=["1d","1wk","1mo"], index=0)
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 14, 1)
lags = st.sidebar.slider("Number of lag features", 3, 30, 10)
test_size_days = st.sidebar.slider("Test set days (most recent)", 30, 365, 90)
model_choice = st.sidebar.selectbox("Model", ["RandomForest", "ARIMA (statsmodels)"])
train_button = st.sidebar.button("Train / Forecast")

# ---------------------------
# Main App Layout
# ---------------------------
st.title("📈 Stock Prediction Dashboard")
st.markdown("Interactive dashboard that pulls data from Yahoo Finance, computes indicators, and trains a simple model to forecast future prices.")

# 1) Download data
with st.spinner("Downloading data..."):
    df_raw = download_data(ticker, period=period, interval=interval)
if df_raw.empty:
    st.error("No data for ticker. Try another ticker symbol or interval.")
    st.stop()

st.subheader(f"{ticker} — Price History")
col1, col2 = st.columns([3,1])
with col1:
    fig = px.line(df_raw.reset_index(), x="Date", y="Adj Close", title=f"{ticker} Adjusted Close Price")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.metric("Latest price", f"${df_raw['Adj Close'][-1]:.2f}", delta=f"{(df_raw['Adj Close'][-1]/df_raw['Adj Close'][-2]-1)*100:.2f}%")

# Show indicators on a combined chart
df_ind = add_technical_indicators(df_raw)
fig2 = go.Figure()
fig2.add_trace(go.Candlestick(x=df_ind.index, open=df_ind['Open'], high=df_ind['High'], low=df_ind['Low'], close=df_ind['Adj Close'], name="OHLC"))
fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind['sma_21'], name="SMA 21", line=dict(width=1)))
fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind['ema_12'], name="EMA 12", line=dict(width=1, dash='dot')))
fig2.update_layout(title=f"{ticker} Price + Indicators", height=500)
st.plotly_chart(fig2, use_container_width=True)

# Prepare ML data
X, y, df_ml = prepare_ml_data(df_raw, horizon=horizon, lags=lags)

# Train/test split (time-based)
split_point = len(df_ml) - test_size_days
if split_point < 50:
    split_point = int(len(df_ml) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
df_test = df_ml.iloc[split_point:]

st.subheader("Dataset & Features")
with st.expander("Show sample features and target"):
    st.dataframe(df_ml.iloc[-200:][['adj_close'] + [c for c in X.columns][:10] + ['target']].tail(50))

st.write(f"Training points: {len(X_train)} — Test points: {len(X_test)}")

# Train / Forecast
if train_button:
    if model_choice == "RandomForest":
        st.info("Training RandomForestRegressor (this may take a few seconds)...")
        # quick baseline hyperparams (you can add RandomizedSearchCV)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        # Predict test
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.success(f"Trained RandomForest — Test MAE: {mae:.4f} — RMSE: {rmse:.4f}")
        # show importance
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
        st.subheader("Feature importances")
        fig_imp = px.bar(importances.reset_index().rename(columns={'index':'feature',0:'importance'}), x='feature', y=0, labels={'0':'importance'})
        st.plotly_chart(fig_imp, use_container_width=True)
        # Plot predictions vs actual (last test_size_days)
        df_plot = pd.DataFrame({'date': df_test.index, 'actual': y_test.values, 'pred': y_pred})
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['actual'], name='Actual'))
        fig_pred.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['pred'], name='Predicted'))
        fig_pred.update_layout(title=f"Test: Actual vs Predicted (horizon={horizon})")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Iterative forecasting for future horizon (predict next N days)
        n_future = st.number_input("Forecast next N days (iterative)", min_value=1, max_value=60, value=7)
        # We'll iteratively produce features for next day
        last_row = df_ml.iloc[-1:].copy()
        future_dates = []
        future_preds = []
        current_row = last_row.copy()
        for i in range(n_future):
            # build feature vector
            feat = {}
            # fill lag_i from previous day's adj_close in current_row
            last_adj = current_row['adj_close'].values[0]
            # we need a vector of lag_1..lag_k: take from current_row (which has lag columns)
            fv = current_row[[c for c in X.columns]].iloc[0].values.reshape(1,-1)
            next_pred = model.predict(fv)[0]
            # append prediction
            next_date = df_ml.index[-1] + pd.Timedelta(days=i+1)
            future_dates.append(next_date)
            future_preds.append(next_pred)
            # update current_row by shifting lag columns (simulate)
            # create a new_row copy
            new_row = current_row.copy()
            # shift lag columns
            for j in range(lags, 1, -1):
                new_row[f'lag_{j}'] = new_row[f'lag_{j-1}'].values
            new_row['lag_1'] = next_pred
            new_row['adj_close'] = next_pred
            # recompute simple rolling features (approx)
            new_row['sma_7'] = new_row['sma_7']  # leave unchanged (approx)
            current_row = new_row
        # show future forecast
        df_future = pd.DataFrame({'date': future_dates, 'forecast': future_preds})
        fig_fut = px.line(df_future, x='date', y='forecast', title="Future iterative forecast")
        st.plotly_chart(fig_fut, use_container_width=True)
        # Export predictions
        csv_buf = df_future.to_csv(index=False).encode('utf-8')
        st.download_button("Download forecast CSV", data=csv_buf, file_name=f"{ticker}_forecast.csv", mime="text/csv")

    elif model_choice == "ARIMA":
        st.info("Fitting a simple ARIMA model on adj_close (this could be slow).")
        # use the entire series for ARIMA; use order (p,d,q) naive defaults (1,1,1)
        try:
            series = df_raw['Adj Close'].dropna()
            # Fit with simple order; for production use auto_arima (pmdarima)
            arima_order = (5,1,0)
            model_ar = ARIMA(series, order=arima_order)
            res = model_ar.fit()
            st.success("ARIMA fitted.")
            # forecast
            n_periods = st.number_input("ARIMA forecast days", min_value=1, max_value=180, value=14)
            fc = res.forecast(steps=n_periods)
            df_fc = pd.DataFrame({'date': pd.date_range(start=series.index[-1]+pd.Timedelta(days=1), periods=n_periods, freq='B'), 'forecast': fc})
            fig_ar = px.line(df_fc, x='date', y='forecast', title=f"ARIMA Forecast ({n_periods} days)")
            st.plotly_chart(fig_ar, use_container_width=True)
            csv_buf = df_fc.to_csv(index=False).encode('utf-8')
            st.download_button("Download ARIMA forecast CSV", data=csv_buf, file_name=f"{ticker}_arima_forecast.csv", mime="text/csv")
        except Exception as e:
            st.error(f"ARIMA failed: {e}")

# If not trained yet show simple historical forecast baseline
else:
    st.info("Hit 'Train / Forecast' in the sidebar to train models and produce predictions. You can tweak model & data settings first.")

# Footer / notes
st.markdown("---")
st.caption("Notes: This dashboard uses simple features and models for demonstration. For production grade forecasting, consider: better feature engineering, hyperparameter search, walk-forward CV, probabilistic forecasting (Prophet / DeepAR), and higher frequency data.")
