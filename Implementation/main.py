import numpy as np
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import io
import base64
import matplotlib.pyplot as plt
import openai

# Function to fetch stock data
def fetch_data(symbol, start_date="2015-01-01", end_date="2023-01-01"):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.ffill()
    stock_data["EMA"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    return stock_data

# Function to fetch company data
def fetch_company_data(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info

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

# Function to generate stock information
def get_stock_info(symbol):
    try:
        stock_info = yf.Ticker(symbol).info
        return stock_info
    except:
        return None

# External CSS styles
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Initialise Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout with improved styling
app.layout = html.Div(
    [
        html.H1(
            "Stock Price Prediction",
            style={"textAlign": "center", "color": "#463f3a"},
        ),
        html.H3(
            "Long Short Term Memory",
            style={"textAlign": "center", "color": "#463f3a", "marginBottom": "20px"},
        ),
        html.Div(
            [
                dcc.Input(
                    id="symbol-input",
                    type="text",
                    value="AAPL",
                    placeholder="Enter a stock symbol",
                    style={
                        "marginBottom": "10px",
                        "padding": "10px",
                        "width": "200px",
                        "borderRadius": "5px",
                        "display": "inline-block",
                    },
                ),
                dcc.DatePickerRange(
                    id="date-picker",
                    start_date="2015-01-01",
                    end_date="2024-01-01",
                    display_format="YYYY-MM-DD",
                    style={
                        "marginBottom": "10px",
                        "marginLeft": "20px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "display": "inline-block",
                    },
                ),
                dcc.Input(
                    id="epoch-input",
                    type="number",
                    value=100,
                    placeholder="Enter Epochs quantity",
                    style={
                        "marginBottom": "10px",
                        "marginTop": "10px",
                        "marginLeft": "20px",
                        "marginRight": "20px",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "display": "inline-block",
                    },
                ),
                html.Button(
                    "Search",
                    id="search-button",
                    n_clicks=0,
                    style={
                        "margin": "auto",
                        "borderRadius": "5px",
                        "background-color": "#bcb8b1",
                        "textAlign": "center",
                        "color": "white",
                        "border": "auto",
                        "font-size": "18px",
                        "display": "inline-block",
                    },
                ),

                # Loading bar whilst the program is running on the terminal
                dcc.Loading(
                    id="loading-output",
                    type="circle",
                    children=[
                        html.Div(id="company-info", style={"marginBottom": "20px"}),
                        dcc.Graph(id="prediction-plot", style={"paddingBottom": "10px"}),
                        html.A(
                            html.Button("Download Graph"),
                            id="download-link",
                            download="prediction_graph.png",
                            href="",
                            target="_blank",
                            style={
                                "marginLeft": "20px",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "background-color": "#bcb8b1",
                                "border": "auto",
                            },
                        ),
                    ],
                ),
            ],

            style={"width": "80%", "margin": "auto", "textAlign": "center"},
        ),
        html.Div(id="stock-info", style={"textAlign": "center", "color": "#463f3a", "fontWeight": "bold"}),
        dcc.Graph(id="loss-plot",
                  style={"width": "80%", "margin": "auto", "paddingBottom": "20px", "paddingTop": "10px"}),
    ],
    style={
        "width": "100%",
        "background-color": "#f4f3ee"
    },
)
# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to ask questions to ChatGPT
def ask_chatgpt(question):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # Choose an appropriate engine
            prompt=f"Ask GPT: {question}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error in asking ChatGPT: {str(e)}"






# Update the company info, plot, and real-time stock info based on user input
@app.callback(
    [
        Output("company-info", "children"),
        Output("stock-info", "children"),
        Output("prediction-plot", "figure"),
        Output("download-link", "href"),
        Output("loss-plot", "figure")
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("epoch-input", "value")
    ]
)
def update_company_and_plot(n_clicks, symbol_input, start_date, end_date, epochs):
    if n_clicks > 0:
        symbol = symbol_input.upper()
        stock_data = fetch_data(symbol, start_date, end_date)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
        model = build_model(sequence_length=25)

     # Ask a question to ChatGPT
        chatgpt_question = f"What can you tell me about {symbol_input} stocks?"
        chatgpt_response = ask_chatgpt(chatgpt_question) 

     # Train the model and get training history
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32)

     # Get the predicted data and actual data
        y_pred = predict_data(model, X_test)
        y_test_actual = inverse_transform(scaler, y_test)
        y_pred_actual = inverse_transform(scaler, y_pred)
        date_range = stock_data.index[-len(y_test):]

     # Plotly graph for Dash
        figure = {
            "data": [
                {"x": stock_data.index, "y": stock_data["EMA"], "type": "line", "name": "Dataset"},
                {"x": date_range, "y": y_test_actual.flatten(), "type": "line", "name": "Actual Data"},
                {"x": date_range, "y": y_pred_actual.flatten(), "type": "line", "name": "Predicted EMA"},
            ],
            "layout": {
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Stock Price"},
                "title": f"{symbol} Stock Price Prediction",
                "legend": {"x": 0, "y": 1}
            }
        }

    # Plotly graph for loss
        loss_figure = {
            "data": [
                {
                    "x": list(range(1, epochs + 1)),
                    "y": history.history['loss'],
                    "type": "line",
                    "name": "Training Loss"
                },
            ],
            "layout": {
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Loss"},
                "title": "Training Loss Over Time",
                "legend": {"x": 0, "y": 1}
            }
        }

     # Update download link
        img_data = io.BytesIO()
        plt.figure(figsize=(8, 6))
        plt.plot(stock_data.index, stock_data["EMA"], label="Dataset")
        plt.plot(date_range, y_test_actual.flatten(), label="Actual Data")
        plt.plot(date_range, y_pred_actual.flatten(), label="Predicted EMA")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{symbol} Stock Price Prediction")
        plt.legend()
        plt.savefig(img_data, format="png")
        img_data.seek(0)
        base64_img = base64.b64encode(img_data.read()).decode("utf-8")
        download_href = f"data:image/png;base64,{base64_img}"

    # Update both outputs
        company_data = fetch_company_data(symbol)
        company_info = [html.P([html.H2(company_data["longName"]),
                                html.P(company_data["longBusinessSummary"])])]

    # Get real-time stock information
        stock_info = get_stock_info(symbol)
        stock_info_display = []
        if stock_info:
            stock_info_display.append(html.P(f"Real-time Stock Price of {stock_info['longName']}"))
            stock_info_display.append(html.P(f"Symbol: {stock_info['symbol']}"))
            stock_info_display.append(html.P(f"Current Price: {stock_info['ask']}"))
            stock_info_display.append(html.P(f"Open: {stock_info['open']}"))
            stock_info_display.append(html.P(f"High: {stock_info['dayHigh']}"))
            stock_info_display.append(html.P(f"Low: {stock_info['dayLow']}"))

        return company_info, stock_info_display, figure, download_href, loss_figure, html.P(f"ChatGPT Response: {chatgpt_response}"),  # Display ChatGPT response 
    else:
    # Return no update for all outputs if no clicks
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

# colour palette 463f3a-8a817c-bcb8b1-f4f3ee-e0afa0

