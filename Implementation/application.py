from dash import dcc, html

from actions import colors

# Set colors used for the webpage
primary1, primary2, primary3, secondary1, secondary2 = colors()

# Creating the webpage
web_app = html.Div(
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
                dcc.Input(
                    id="sequence-input",
                    type="number",
                    value=25,
                    placeholder="Enter sequence quantity",
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
                dcc.Input(
                    id="batch-input",
                    type="number",
                    value=32,
                    placeholder="Enter batch quantity",
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
        
        # Toggle dark mode
        html.Div(id="output", children=[
        html.Button("Toggle Dark Mode", id="toggle-dark-mode"),
        html.Div(id="output")
    ]),
    
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


# Return the web page for the application
def app():
    return web_app
