import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# we should put the data processing in like processing.py file
symbol = input("Enter a stock symbol: ")
# symbol = "TSLA"

# fetch historical data for processing?
stock_data = yf.download(symbol, start="2020-01-01", end="2023-01-01")
stock_data = stock_data.ffill()  # preproc

# scales data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))
train_size = int(len(scaled_data) * 0.8)  # split data for test/train
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)


# creates sequences
sequence_length = 10

X_train = create_sequences(train_data, sequence_length)
X_test = create_sequences(test_data, sequence_length)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# preps target
y_train = train_data[sequence_length:]
y_test = test_data[sequence_length:]

# build rnn model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer="RMSprop", loss="mean_squared_error")

# train model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# predict the test data
y_pred = model.predict(X_test)


# inverse transform the scaled data to the original scale
y_test_actual = scaler.inverse_transform(y_test)
y_pred_actual = scaler.inverse_transform(y_pred)

mse = np.sqrt(mean_squared_error(y_test_actual,y_pred_actual))
print(f"Root Mean Squared Error is: {mse}")


# create a time series index for the test data
date_range = stock_data.index[-len(y_test):]

# plot the actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(date_range, y_test_actual, label="Actual Price")
plt.plot(date_range, y_pred_actual, label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{symbol} Stock Price Prediction")
plt.legend()
plt.show()
