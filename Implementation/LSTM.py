import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

def fetch_data(symbol, start_date="2020-01-01", end_date="2023-01-01"):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.ffill()
    stock_data['EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    return stock_data

def preprocess_data(stock_data, sequence_length=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data["EMA"].values.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train = create_sequences(train_data, sequence_length)
    X_test = create_sequences(test_data, sequence_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = train_data[sequence_length:]
    y_test = test_data[sequence_length:]

    return X_train, X_test, y_train, y_test, scaler

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def build_model(sequence_length):
    model = Sequential()
    model.add(LSTM(150, activation="relu", input_shape=(sequence_length, 1)))
    # model.add(Dropout(0.2))  # dropout layer to prevent overfitting
    model.add(Dense(1))
    model.compile(optimizer="RMSprop", loss="mean_squared_error")
    return model

def train_model(model, X_train, y_train, epochs=150, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

def predict_data(model, X_test):
    return model.predict(X_test)

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def plot_predictions(stock_data, date_range, y_test_actual, y_pred_actual, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['EMA'], label="Actual dataset (Entire Dataset)", color="gray", linewidth=1)
    plt.plot(date_range, y_test_actual, label="Actual (Test Data)", linewidth=1)
    plt.plot(date_range, y_pred_actual, label="Predicted EMA", linewidth=1)

    plt.text(date_range[-1] + pd.DateOffset(days=3), y_test_actual[-1][0] + 2,
             f'Actual: {y_test_actual[-1][0]:.2f}', ha='left', va='center', fontsize=10)

    plt.text(date_range[-1] + pd.DateOffset(days=3), y_pred_actual[-1][0] - 2,
             f'Predicted: {y_pred_actual[-1][0]:.2f}', ha='right', va='center', fontsize=10)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{symbol} Stock Price Prediction")
    plt.legend()
    plt.show()

def run(symbol, start_date="2020-01-01", end_date="2023-01-01", sequence_length=10, epochs=100, batch_size=32):
    stock_data = fetch_data(symbol, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, sequence_length)
    model = build_model(sequence_length)
    train_model(model, X_train, y_train, epochs, batch_size)
    y_pred = predict_data(model, X_test)
    y_test_actual = inverse_transform(scaler, y_test)
    y_pred_actual = inverse_transform(scaler, y_pred)
    date_range = stock_data.index[-len(y_test):]
    plot_predictions(stock_data, date_range, y_test_actual, y_pred_actual, symbol)

# usage:
symbol_input = input("Enter a stock symbol: ")
run(symbol_input)
