import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from ml_model import run_model

st.title("Advanced Stock Prediction & Trading Strategy")

# -------------------------
# User Inputs
# -------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Click here to Run Model"):

    with st.spinner("Running model..."):

        df, y_test, y_pred, rmse, mae, r2 = run_model(ticker, start, end)

        # -------------------------
        # Metrics
        # -------------------------
        st.subheader("Model Performance")
        st.write("RMSE:", rmse)
        st.write("MAE:", mae)
        st.write("R² Score:", r2)

        # -------------------------
        # Prediction Chart
        # -------------------------
        st.subheader("Predicted vs Actual Prices")

        fig = plt.figure(figsize=(12,6))
        plt.plot(y_test, label="Actual Price")
        plt.plot(y_pred, label="Predicted Price")
        plt.legend()
        st.pyplot(fig)

        # -------------------------
        # Strategy
        # -------------------------
        st.subheader("Backtesting Strategy")

        signals = (y_pred.flatten() > y_test.flatten()).astype(int)

        returns = df['Close'].pct_change().iloc[-len(signals):]
        strategy_returns = returns.values * signals

        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_market = (1 + returns.values).cumprod()

        fig2 = plt.figure(figsize=(12,6))
        plt.plot(cumulative_strategy, label="ML Strategy")
        plt.plot(cumulative_market, label="Buy & Hold")
        plt.legend()
        st.pyplot(fig2)

        st.success("Model & Strategy Evaluation Complete!")