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
import os
from openai import OpenAI

# Colour palette 463f3a-8a817c-bcb8b1-f4f3ee-e0afa0
primary1 = "#bcb8b1"
primary2 = "#8a817c"
primary3 = "#463f3a"
secondary1 = "#f4f3ee"
secondary2 = "#e0afa0"

# Set configs
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: You currently do not have an OpenAI API key set. Please check "
          "the README.md file in the project directory for more information.")


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


# Function make a call to the OpenAI API to generate a response
def generate_response(request):
    context = "You are a stock analyst assistant who provides stock information to the user."
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": request}
        ],
    )
    return completion.choices[0].message.content


# External CSS styles
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Initialise Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout with improved styling
app.layout = html.Div(
    [
        # Header
        html.H1(
            "Stock Price Prediction",
            style={"textAlign": "center", "color": primary3, "font-weight": "bold"},
        ),
        html.H3(
            "Long Short Term Memory",
            style={"textAlign": "center", "color": primary3, "marginBottom": "10px"},
        ),

        # Main divider for user area
        html.Div(
            [
                # User inputs for analysis
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
                        "background": primary3,
                        "textAlign": "center",
                        "color": secondary1,
                        "border": "auto",
                        "font-size": "15px",
                        "display": "inline-block",
                    },
                ),

                # Display information about the company to the user
                dcc.Loading(
                    type="circle",
                    children=[html.Div(id="company-info", style={"marginBottom": "20px", "min-height": "50px"})],
                ),

                # Display the prediction graph to the user
                dcc.Loading(
                    id="loading-output",
                    type="circle",
                    children=[
                        dcc.Graph(id="prediction-plot", style={"paddingBottom": "10px"}),
                        html.A(
                            html.Button("Download Graph", style={
                                "borderRadius": "5px",
                                "background": primary3,
                                "color": secondary1}, ),
                            id="download-link",
                            download="prediction_graph.png",
                            href="",
                            target="_blank",
                        ),
                    ],
                ),
            ],

            style={"width": "80%", "margin": "auto", "textAlign": "center"},
        ),

        # Chat Interface
        html.Div([
            html.Div(id="chat-container", children=[
                html.Div(children=[
                    html.Span(children="Welcome to the Stock Information Chat!", style={"font-weight": "bold"})
                ]),
            ], style={"background": primary2, "color": secondary1, "borderRadius": "5px",
                      "padding": "10px", "margin": "10px 0"}),
            dcc.Loading(
                type="circle",
                children=[
                    html.Div(id="chat-output-text",
                             style={"background": "#e6e6e6",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "margin": "10px 0",
                                    "height": "300px",
                                    "overflowY": "scroll"}), ]
            ),
            html.Div(children=[
                dcc.Textarea(id="chat-input-text", value="", placeholder="Type your message...",
                             style={"width": "100%", "height": "100px"}),
                html.Button("Submit", id="chat-submit-button",
                            style={"width": "95%",
                                   "background": primary3,
                                   "color": secondary1,
                                   "margin": "auto",
                                   "border": "none",
                                   "cursor": "pointer",
                                   "textAlign": "center"})
            ])
        ],
            style={"width": "80%", "margin": "auto", "marginTop": "20px", "paddingBottom": "10px",
                   "background": primary1, "borderRadius": "15px", "textAlign": "center"}
        ),

        # Display extra information for user
        html.Div(id="rt-stock-info", style={"paddingTop": "20px"}),
        html.Div(
            id="stock-info",
            children=[],
            style={"width": "80%", "display": "grid", "grid-template-columns": "repeat(2, 1fr)", "gap": "5px",
                   "textAlign": "center", "margin": "auto", "paddingBottom": "20px", "paddingTop": "5px"},
        ),
        dcc.Loading(
            type="circle",
            children=[
                dcc.Graph(id="loss-plot",
                          style={"width": "80%", "margin": "auto", "paddingBottom": "10px", "paddingTop": "10px"})]
        ),
    ],
    style={
        "width": "100%",
        "background": secondary1
    },
)


# Update company information
@app.callback(
    [
        Output("company-info", "children"),
        Output("chat-input-text", "value")
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value")
    ]
)
def update_company(n_clicks, symbol_input):
    if n_clicks > 0:
        print("Action: Updating company information.")
        # Fetch information about the company
        company_data = fetch_company_data(symbol_input)

        # Summarise long summary
        try:
            summary = generate_response("Summarise in a short paragraph: " + company_data["longBusinessSummary"])
            company_info = [html.P([html.H2(company_data["longName"]),
                                    html.P(summary)])]
            print("Output: Returning OpenAPI call results: Summary.")
        except:
            print("Request Denied: OpenAI API request cannot be completed. Please check "
                  "the README.md file in the project directory for more information.")
            company_info = [html.P([html.H2(company_data["longName"])])]

        print("Action: Updating response in ChatBot input.")
        # Prompt for stock symbol
        promt = ("Evaluate important events that caused the stock price of " + company_data["longName"]
                 + " (" + symbol_input + ") to change.")

        # Return updates
        if company_info:
            return company_info, promt
    else:
        return dash.no_update, dash.no_update


# Update the company info, plot, and real-time stock info based on user input
@app.callback(
    [
        Output("prediction-plot", "figure"),
        Output("download-link", "href"),
        Output("rt-stock-info", "children"),
        Output("stock-info", "children"),
        Output("loss-plot", "figure"),
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("epoch-input", "value")
    ]
)
def update_plot_and_info(n_clicks, symbol_input, start_date, end_date, epochs):
    if n_clicks > 0:
        print("Action: Processing stock data.")
        # Fetching and preprocessing data
        symbol = symbol_input.upper()
        stock_data = fetch_data(symbol, start_date, end_date)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
        model = build_model(sequence_length=25)

        # Train the model and get training history
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32)

        # Get the predicted data and actual data
        y_pred = predict_data(model, X_test)
        y_test_actual = inverse_transform(scaler, y_test)
        y_pred_actual = inverse_transform(scaler, y_pred)
        date_range = stock_data.index[-len(y_test):]

        print("Action: Plotting graphs using prediction data.")
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
                    "y": history.history["loss"],
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

        print("Action: Updating real-time stock information.")
        # Get real-time stock information
        stock_info = get_stock_info(symbol)
        current_stock = html.Div(f"Real-time Stock Price of {stock_info['longName']}",
                                 style={"paddingTop": "5px", "paddingBottom": "5px", "background": primary2,
                                        "color": secondary1, "borderRadius": "5px", "textAlign": "center",
                                        "width": "80%", "margin": "auto", "font-weight": "bold"})

        stock_info_display = []
        if stock_info:
            styles = {"padding": "10px", "background": primary1, "borderRadius": "5px"}
            stock_info_display = [
                html.Div(f"Current Price: {stock_info['ask']}", style=styles),
                html.Div(f"Open: {stock_info['open']}", style=styles),
                html.Div(f"Low: {stock_info['dayLow']}", style=styles),
                html.Div(f"High: {stock_info['dayHigh']}", style=styles),
            ]

        # Return updates
        return figure, download_href, current_stock, stock_info_display, loss_figure
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Used for ChatGPT ChatBox
@app.callback(
    Output("chat-output-text", "children"),
    [Input("chat-submit-button", "n_clicks")],
    [State("chat-input-text", "value")]
)
def user_chat(n_clicks, input_text):
    if n_clicks is not None:
        print("Action: Generating a response from OpenAI API.")
        # Send user input to function to call API
        try:
            output = generate_response(input_text)
            print("Output: Returning OpenAI API call result: ChatBox.")
            return output
        except:
            print("Request Denied: OpenAI API request cannot be completed. Please check "
                  "the README.md file in the project directory for more information.")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

# colour palette 463f3a-8a817c-bcb8b1-f4f3ee-e0afa0
