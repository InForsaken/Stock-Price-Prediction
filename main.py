import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# we should put the data processing in like processing.py file
symbol = "AAPL"

# fetch historical data for processing?
apple_data = yf.download(symbol, start="2020-01-01", end="2023-01-01")
apple_data = apple_data.ffill() # preproc

# scales data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(apple_data["Close"].values.reshape(-1, 1))
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
model.add(SimpleRNN(50, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer="RMSProp", loss="mean_squared_error")

# train model
model.fit(X_train, y_train, epochs=50, batch_size=32)