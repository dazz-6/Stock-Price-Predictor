import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.title("Advanced Stock Prediction & Trading Strategy")

# -------------------------
# User Inputs
# -------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

df = load_data(ticker, start, end)

if df.empty:
    st.error("No data found.")
    st.stop()

# -------------------------
# Feature Engineering
# -------------------------
df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

# RSI
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26

df.dropna(inplace=True)

features = ['Close','Volume','MA50','MA200','EMA20','RSI','MACD']
data = df[features]

# -------------------------
# Scaling
# -------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------
# Sequence Creation
# -------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict Close
    return np.array(X), np.array(y)

seq_length = 100
X, y = create_sequences(scaled_data, seq_length)

# -------------------------
# Train/Val/Test Split (60/20/20)
# -------------------------
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# -------------------------
# Model
# -------------------------
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

with st.spinner("Training model..."):
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)

# Inverse scaling
close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']])

y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = close_scaler.inverse_transform(y_pred)

# -------------------------
# Metrics
# -------------------------
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

st.subheader("Model Performance")
st.write("RMSE:", rmse)
st.write("MAE:", mae)
st.write("R² Score:", r2)

# -------------------------
# Plot Predictions
# -------------------------
st.subheader("Predicted vs Actual Prices")

fig = plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="Actual Price")
plt.plot(y_pred_inv, label="Predicted Price")
plt.legend()
st.pyplot(fig)

# -------------------------
# Trading Strategy
# -------------------------
st.subheader("Backtesting Strategy")

signals = (y_pred_inv.flatten() > y_test_inv.flatten()).astype(int)

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