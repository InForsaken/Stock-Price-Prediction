# Stock Price Prediction

## Table of Contents
- [Overview](#Overview)
    - [Description](#Description)
    - [Features](#Features) 
- [Installation](#Installation)
    - [Prerequisites](#Prerequisites)
    - [How to run](#How-to-run)
    - [Usage](#Usage)

### Overview

## Description

This project is a web-based stock price prediction application built using Dash, Keras, and other Python libraries. The application allows users to input a stock symbol, select a date range, and specify the number of epochs for training an LSTM (Long Short-Term Memory) model. The predicted stock prices are visualized along with additional information such as company details and real-time stock information.

## Features

- **Stock Data Fetching:** Utilizes the Yahoo Finance API (`yfinance`) to fetch historical stock data.
- **Data Preprocessing:** Performs data preprocessing, including the calculation of Exponential Moving Average (EMA) and scaling of data.
- **LSTM Model Training:** Constructs and trains an LSTM model using the Keras library.
- **Interactive Web Interface:** Provides a user-friendly interface with Dash for inputting parameters and viewing predictions.
- **Real-time Stock Information:** Retrieves real-time stock information using the Yahoo Finance API.

### Installation

## Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install numpy yfinance dash keras scikit-learn matplotlib
```

## How to run

To run the application, simply run the main script:

```
python main.py
```

The application will start a local server and you can view the application by navigating to ```http://127.0.0.1:8050/``` in your web browser.

## Usage

1. Enter a stock symbol in the input field.
2. Select a date range using the date picker.
3. Specify the number of epochs for model training.
4. Click the "Search" button to view the stock price predictions.
