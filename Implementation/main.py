import base64
import io

import dash
import matplotlib.pyplot as plt
from dash import html
from dash.dependencies import (Input, Output, State)

import application
from actions import (colors, fetch_data, fetch_company_data, preprocess_data, build_model,
                     predict_data, inverse_transform, get_stock_info, generate_response,calculate_historical_volatility)

# Set colors used for the webpage
primary1, primary2, primary3, secondary1, secondary2 = colors()

# External CSS styles
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Initialise Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout with improved styling
app.layout = application.web_app


# Update company information
@app.callback(
    [
        Output("company-info", "children"),
        Output("chat-input-text", "value"),
        Output("rt-stock-info", "children"),
        Output("stock-info", "children")
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
        print("Request: Generating a summary from OpenAI API - Summary")
        try:
            summary = generate_response("Summarise in a short paragraph: " + company_data["longBusinessSummary"])
            company_info = [html.P([html.H2(company_data["longName"]),
                                    html.P(summary)])]
            print("Request Accepted: Returning OpenAPI call results - Summary.")
        except:
            company_info = [html.P([html.H2(company_data["longName"]),
                                    html.P(company_data["longBusinessSummary"])])]
            print("Request Denied: OpenAI API request cannot be completed. Please check "
                  "the README.md file in the project directory for more information.")

        print("Action: Updating response in ChatBot input.")
        # Prompt for stock symbol
        promt = ("Evaluate important events that caused the stock price of " + company_data["longName"]
                 + " (" + symbol_input + ") to change.")

        print("Action: Updating real-time stock information.")
        # Get real-time stock information
        stock_info = get_stock_info(symbol_input)
        current_stock = html.Div(f"Real-time Stock Price of {stock_info['longName']}",
                                 style={"paddingTop": "5px",
                                        "paddingBottom": "5px",
                                        "background": primary2,
                                        "color": secondary1,
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "width": "80%",
                                        "margin": "auto",
                                        "font-weight": "bold"})

        stock_info_display = []
        if stock_info:
            styles = {"padding": "10px", "background": primary1, "borderRadius": "5px"}
            stock_info_display = [
                html.Div(f"Current Price: {stock_info['ask']:.2f}", style=styles),
                html.Div(f"Open: {stock_info['open']:.2f}", style=styles),
                html.Div(f"Low: {stock_info['dayLow']:.2f}", style={**styles, "color": "#ff0000"}),
                html.Div(f"High: {stock_info['dayHigh']:.2f}", style={**styles, "color": "#008000"})
            ]

        # Return updates
        if company_info:
            return company_info, promt, current_stock, stock_info_display,
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Update the company info, plot, and real-time stock info based on user input
@app.callback(
    [
        Output("prediction-plot", "figure"),
        Output("download-link", "href"),
        Output("loss-plot", "figure"),
        Output("historical-volatility-plot", "figure")  # Add this output
    ],
    [Input("search-button", "n_clicks")],
    [
        State("symbol-input", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("epoch-input", "value"),
        State("sequence-input", "value"),
        State("batch-input", "value")
    ]
)
def update_plot_and_info(n_clicks, symbol_input, start_date, end_date, epochs, sequence, batch):
    if n_clicks > 0:
        # Fetching and preprocessing data
        symbol = symbol_input.upper()
        stock_data = fetch_data(symbol, start_date, end_date)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
        model = build_model(sequence_length=sequence)

        # Train the model and get training history
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch)

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
                {"x": date_range, "y": y_pred_actual.flatten(), "type": "line", "name": "Predicted EMA"}
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
                }
            ],
            "layout": {
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Loss"},
                "title": "Training Loss Over Time",
                "legend": {"x": 0, "y": 1}
            }
        }

        # Calculate historical volatility
        historical_volatility = calculate_historical_volatility(stock_data)

        # Plotly graph for historical volatility
        volatility_figure = {
            "data": [
                {"x": historical_volatility.index, "y": historical_volatility, "type": "line", "name": "Historical Volatility"}
            ],
            "layout": {
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Volatility"},
                "title": f"{symbol} Historical Volatility",
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

        # Return updates
        return figure, download_href, loss_figure, volatility_figure
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update



# Used for ChatGPT ChatBox
@app.callback(
    Output("chat-output-text", "children"),
    [Input("chat-submit-button", "n_clicks")],
    [State("chat-input-text", "value")]
)
def user_chat(n_clicks, input_text):
    if n_clicks is not None:
        # Send user input to function to call API
        print("Request: Generating a response from OpenAI API - ChatBox")
        try:
            output = generate_response(input_text)
            print("Request Accepted: Returning OpenAI API call result - ChatBox.")
        except:
            output = ("Request Denied: OpenAI API request cannot be completed. Please check "
                      "the README.md file in the project directory for more information.")
            print(output)
        return output


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)