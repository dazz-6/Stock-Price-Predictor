import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

import os


# -------------------------
# Load Data
# -------------------------
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)


# -------------------------
# Feature Engineering
# -------------------------
def add_features(df):
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    df.dropna(inplace=True)
    return df


# -------------------------
# Create Sequences
# -------------------------
def create_sequences(data, seq_length=100):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# -------------------------
# Train Model (only once)
# -------------------------
def train_and_save_model():

    print("Training model...")

    df = load_data("AAPL", "2015-01-01", "2024-12-31")
    df = add_features(df)

    features = ['Close','Volume','MA50','MA200','EMA20','RSI','MACD']
    data = df[features]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data)

    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]

    model = Sequential()

    model.add(LSTM(64, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.save("model.keras")

    print("✅ Model saved as model.keras")


# -------------------------
# Run Pipeline (FAST)
# -------------------------
def run_model(ticker, start, end):

    df = load_data(ticker, start, end)
    df = add_features(df)

    features = ['Close','Volume','MA50','MA200','EMA20','RSI','MACD']
    data = df[features]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data)

    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]

    # ✅ Load model instead of training
    if not os.path.exists("model.keras"):
        raise FileNotFoundError("❌ model.keras not found. Run training first.")

    model = load_model("model.keras")

    y_pred = model.predict(X_test)

    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['Close']])

    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred_inv = close_scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    return df, y_test_inv, y_pred_inv, rmse, mae, r2


# -------------------------
# Run once manually
# -------------------------
if __name__ == "__main__":
    train_and_save_model()