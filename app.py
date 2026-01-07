import streamlit as st
import pickle
import matplotlib.pyplot as plt

from src.data_loader import load_stock_data
from src.preprocessing import preprocess

st.set_page_config(page_title="Stock Market Prediction", layout="wide")

st.title("📈 Stock Market Trend Prediction Dashboard")

# Load data
df = load_stock_data()
df = preprocess(df)

# Sidebar
company = st.sidebar.selectbox("Select Company", df["Company"].unique())
company_df = df[df["Company"] == company]

# Plot
st.subheader(f"Price Trend for {company}")
fig, ax = plt.subplots()
ax.plot(company_df["Date"], company_df["Close"], label="Close Price")
ax.plot(company_df["Date"], company_df["MA_10"], label="MA 10")
ax.plot(company_df["Date"], company_df["MA_20"], label="MA 20")
ax.legend()
st.pyplot(fig)

# Prediction
st.subheader("📊 Predict Next-Day Trend")

latest = company_df.iloc[-1][
    ["Open", "High", "Low", "Close", "Volume", "MA_10", "MA_20"]
].values.reshape(1, -1)

with open("model/stock_model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(latest)[0]

if prediction == 1:
    st.success("📈 Prediction: Stock price likely to go UP")
else:
    st.error("📉 Prediction: Stock price likely to go DOWN")
