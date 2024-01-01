import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

# Function to fetch stock data
def fetch_data(symbol, start_date="2015-01-01", end_date="2023-01-01"):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.ffill()
    stock_data['EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    return stock_data

# Function to preprocess data
def preprocess_data(stock_data, sequence_length=25):
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

# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Function to build LSTM model
def build_model(sequence_length):
    model = Sequential()
    model.add(LSTM(150, activation="relu", input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer="RMSprop", loss="mean_squared_error")
    return model

# Function to train the model
def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Function to predict data
def predict_data(model, X_test):
    return model.predict(X_test)

# Function to inverse transform scaled data
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Stock Price Prediction with LSTM"),
    html.Div([
    dcc.Input(id='symbol-input1', type='text', value='TSLA', placeholder='Enter a stock symbol 1'),
    dcc.Input(id='symbol-input2', type='text', value='AAPL', placeholder='Enter a stock symbol 2'),
    ]),

html.Div([
dcc.Graph(id='stock1-plot',className='columns'),
dcc.Graph(id='stock2-plot',className='columns')],
className='row')
])

# Update the graph based on user input
@app.callback(
    [Output('stock1-plot', 'figure'),Output('stock2-plot', 'figure')],
    [Input('symbol-input1','value'), Input('symbol-input2','value')]
)

def update_plot(symbol_input1, symbol_input2):
    symbols = [symbol_input1.upper(), symbol_input2.upper()]
    figures=[]
    for symbol in symbols:
     stock_data = fetch_data(symbol)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
    model = build_model(sequence_length=25)
    train_model(model, X_train, y_train)
    y_pred = predict_data(model, X_test)
    y_test_actual = inverse_transform(scaler, y_test)
    y_pred_actual = inverse_transform(scaler, y_pred)
    date_range = stock_data.index[-len(y_test):]

    # Plotly graph for Dash
    figure = {
        'data': [
            {'x': stock_data.index, 'y': stock_data['EMA'], 'type': 'line', 'name': f'actual{symbol}'},
            {'x': date_range, 'y': y_test_actual.flatten(), 'type': 'line', 'name': f'actual{symbol} (test data)'},
            {'x': date_range, 'y': y_pred_actual.flatten(), 'type': 'line', 'name': f'Predicted {symbol}'},
        ],
        'layout': {
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Stock Price'},
            'title': f"{symbol} Stock Price Prediction",
            'legend': {'x': 0, 'y': 1}
        }
    }
    figures.append(figure)

    return figures[0], figures[1]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)