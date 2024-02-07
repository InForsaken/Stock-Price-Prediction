# Stock Price Prediction

## Table of Contents
- [Overview](#Overview)
    - [Description](#Description)
    - [Features](#Features)
    - [Functionality](#Functionality)
- [Installation](#Installation)
    - [Prerequisites](#Prerequisites)
    - [How to run](#How-to-run)
    - [Usage](#Usage)
    - [Notes](#Notes)

## Overview

### Description
This project is a web-based stock price prediction application built using Dash, Keras, and other Python libraries. The application allows users to input a stock symbol, select a date range, and specify the number of epochs for training an LSTM (Long Short-Term Memory) model. The predicted stock prices are visualized along with additional information such as company details and real-time stock information.

### Features
- **Stock Data Fetching:** Utilizes the Yahoo Finance API (`yfinance`) to fetch historical stock data.
- **Data Preprocessing:** Performs data preprocessing, including the calculation of Exponential Moving Average (EMA) and scaling of data.
- **LSTM Model Training:** Constructs and trains an LSTM model using the Keras library.
- **Interactive Web Interface:** Provides a user-friendly interface with Dash for inputting parameters and viewing predictions.
- **Real-time Stock Information:** Retrieves real-time stock information using the Yahoo Finance API.

### Functionality
- **Fetching Data** 
    - **fetch_data(symbol, start_date, end_date):** Fetches historical stock data using Yahoo Finance API for the specified stock symbol and date range.
    - **fetch_company_data(symbol):** Fetches company information for the specified stock symbol.

- **Preprocessing Data**
    - **preprocess_data(stock_data, sequence_length):** Preprocesses the fetched data by scaling and splitting it into training and testing sets.

- **Model Building and Training**
    - **build_model(sequence_length):** Builds an LSTM model with the specified sequence length.
    - **train_model(model, X_train, y_train, epochs, batch_size):** Trains the LSTM model on the training data.

- **Prediction and Visualization**
    - **predict_data(model, X_test):** Generates predictions based on the trained model.
    - **inverse_transform(scaler, data):** Inverse transforms scaled data to its original form.
    - **get_stock_info(symbol):** Shows real-time stock information for the specified symbol.

## Installation

### Prerequisites
Make sure you have the following Python packages installed:
- numpy
- yfinance
- dash
- keras
- scikit-learn
- matplotlib
- openai

Install the packages using:
```bash
pip install numpy yfinance dash keras scikit-learn matplotlib openai
```
Set your OpenAI API key:
```bash
setx OPENAI_API_KEY "key"
```

### How to run
To run the application, simply run the main script:
```python
py main.py
```
The application will start a local server and you can view the application by navigating to ```http://127.0.0.1:8050/``` in your web browser.


### Usage
Dash Application
The Dash application provides a user interface for interacting with the model:
1. Enter a stock symbol in the input field.
2. Select a date range using the date picker.
3. Specify the number of epochs for model training.
4. Click the "Search" button to view the stock price predictions.

### Notes
Ensure a stable internet connection to fetch real-time stock information.
Some stock symbols may not be available or may have limited past data.
Adjusting the number of  epochs may affect the model's accuracy and training time.
