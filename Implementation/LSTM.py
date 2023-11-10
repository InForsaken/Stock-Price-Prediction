import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

class StockPricePredictor:

    def __init__(self, symbol, start_date="2020-01-01", end_date="2023-01-01", sequence_length=10):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        stock_data = stock_data.ffill()
        stock_data['EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        return stock_data

    def preprocess_data(self, stock_data):
        scaled_data = self.scaler.fit_transform(stock_data["EMA"].values.reshape(-1, 1))
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        X_train = self.create_sequences(train_data)
        X_test = self.create_sequences(test_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        y_train = train_data[self.sequence_length:]
        y_test = test_data[self.sequence_length:]

        return X_train, X_test, y_train, y_test

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(150, activation="relu", input_shape=(self.sequence_length, 1)))
        # model.add(Dropout(0.2))  # dropout layer to prevent overfitting
        model.add(Dense(1))
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        # model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train_model(self, X_train, y_train, epochs=150, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_data(self, X_test):
        return self.model.predict(X_test)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def plot_predictions(self, stock_data, date_range, y_test_actual, y_pred_actual):
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['EMA'], label="Actual dataset (Entire Dataset)", color="gray", linewidth=1)
        plt.plot(date_range, y_test_actual, label="Actual (Test Data)", linewidth=1)
        plt.plot(date_range, y_pred_actual, label="Predicted EMA", linewidth=1)

        plt.text(date_range[-1] + pd.DateOffset(days=3), y_test_actual[-1][0] + 2,
                 f'Actual: {y_test_actual[-1][0]:.2f}', ha='left', va='center', fontsize=10)

        plt.text(date_range[-1] + pd.DateOffset(days=3), y_pred_actual[-1][0] - 2,
                 f'Predicted: {y_pred_actual[-1][0]:.2f}', ha='left', va='center', fontsize=10)

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{self.symbol} Stock Price Prediction")
        plt.legend()
        plt.show()

    def run(self, epochs=100, batch_size=32):
        stock_data = self.fetch_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(stock_data)
        self.model = self.build_model()
        self.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size)
        y_pred = self.predict_data(X_test)
        y_test_actual = self.inverse_transform(y_test)
        y_pred_actual = self.inverse_transform(y_pred)
        date_range = stock_data.index[-len(y_test):]
        self.plot_predictions(stock_data, date_range, y_test_actual, y_pred_actual)


# usage:
symbol_input = input("Enter a stock symbol: ")
predictor = StockPricePredictor(symbol=symbol_input)
predictor.run()
