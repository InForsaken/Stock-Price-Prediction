import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class StockPricePredictor:
    
    def __init__(self, symbol, start_date="2015-01-01", end_date="2023-01-01", sequence_length=10):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length

    def fetch_stock_data(self):
        try:
            # fetch historical data for processing
            stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            stock_data = stock_data.ffill()  # preproc
            return stock_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def scale_data(self, data):
        # scales data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        return scaler, scaled_data

    def create_sequence_sets(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def preprocess_data(self, stock_data):
        if stock_data is None:
            return None, None, None, None, None

        # scales data
        scaler, scaled_data = self.scale_data(stock_data["Close"].values)
        train_size = int(len(scaled_data) * 0.8)  # split data for test/train

        # creates sequences
        X_train = self.create_sequence_sets(scaled_data[:train_size])
        X_test = self.create_sequence_sets(scaled_data[train_size:])
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # preps target
        y_train = scaled_data[self.sequence_length:train_size]
        y_test = scaled_data[self.sequence_length + train_size:]

        return scaler, X_train, X_test, y_train, y_test

    def build_model(self):
        # build LSTM model with dropout layers
        model = Sequential()
        model.add(LSTM(100, activation="relu", input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))  # dropout layer to prevent overfitting
        model.add(Dense(1))
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        return model

    def train_model(self, model, X_train, y_train, epochs=100, batch_size=32):
        # train model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return model, history

    def predict_prices(self, model, X_test):
        # predict the test data
        y_pred = model.predict(X_test)
        return y_pred

    def evaluate_model(self, scaler, y_test_actual, y_pred_actual):
        # inverse transform the scaled data to the original scale
        y_test_actual = scaler.inverse_transform(y_test_actual)
        y_pred_actual = scaler.inverse_transform(y_pred_actual)

        # evaluate the model
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        return mse, mae

    def plot_predictions(self, date_range, y_test_actual, y_pred_actual):
        # create a time series index for the test data
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, y_test_actual, label="Actual Price")
        plt.plot(date_range, y_pred_actual, label="Predicted Price")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{self.symbol} Stock Price Prediction")
        plt.legend()
        plt.show()

    def plot_training_loss(self, history):
        # plot the training loss over epochs
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def run(self, epochs=100, batch_size=32):
        stock_data = self.fetch_stock_data()
        
        if stock_data is None:
            return

        scaler, X_train, X_test, y_train, y_test = self.preprocess_data(stock_data)

        model = self.build_model()
        model, history = self.train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

        y_pred = self.predict_prices(model, X_test)

        date_range = stock_data.index[-len(y_test):]
        mse, mae = self.evaluate_model(scaler, y_test, y_pred)
        self.plot_predictions(date_range, y_test, y_pred)
        self.plot_training_loss(history)

# Usage
symbol = input("Enter a stock symbol: ")
predictor = StockPricePredictor(symbol)
predictor.run()