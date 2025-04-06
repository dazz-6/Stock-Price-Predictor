import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model  # type: ignore

import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler  # type: ignore

# App title
st.title("Stock Trend Prediction")

# Input section
user_input = st.text_input("Enter Stock Ticker", "AAPL")
start = '2015-01-01'
end = '2024-12-31'

# Download stock data
@st.cache_data

def load_data(ticker):
    return yf.download(ticker, start=start, end=end)

df = load_data(user_input)

# Error handling for empty data
if df.empty:
    st.error("No stock data found for the selected symbol and time period. Try a different one.")
    st.stop()

# Data summary
st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# Plot 1: Closing price
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

# Plot 2: 100-day MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA')
plt.plot(df['Close'], label='Closing Price')
plt.legend()
st.pyplot(fig)

# Plot 3: 100-day and 200-day MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)

# Splitting data
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
st.write("Training data shape:", data_training.shape)
st.write("Training data preview:", data_training.head())

# Scaling training data
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

x_train, y_train = [], []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load model safely
if not os.path.exists('my_model.keras'):
    st.error("Model file not found!")
    st.stop()
model = load_model('my_model.keras')

# Prepare test data
past_100_days = data_training.tail(100)
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_data)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict and inverse scale
y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot prediction vs original
st.subheader('Predictions vs Original')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)