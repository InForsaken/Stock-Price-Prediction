import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog


def fetch_data(symbol, start_date="2020-01-01", end_date="2023-01-01"):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data = stock_data.ffill()
    stock_data['EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    return stock_data

def preprocess_data(stock_data, sequence_length=10):
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

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def build_model(sequence_length):
    model = Sequential()
    model.add(LSTM(150, activation="relu", input_shape=(sequence_length, 1)))
    # model.add(Dropout(0.2))  # dropout layer to prevent overfitting
    model.add(Dense(1))
    model.compile(optimizer="RMSprop", loss="mean_squared_error")
    return model

def train_model(model, X_train, y_train, epochs=150, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

def predict_data(model, X_test):
    return model.predict(X_test)

def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

def plot_predictions(stock_data, date_range, y_test_actual, y_pred_actual, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['EMA'], label="Actual dataset (Entire Dataset)", color="gray", linewidth=1)
    plt.plot(date_range, y_test_actual, label="Actual (Test Data)", linewidth=1)
    plt.plot(date_range, y_pred_actual, label="Predicted EMA", linewidth=1)

    plt.text(date_range[-1] + pd.DateOffset(days=3), y_test_actual[-1][0] + 2,
             f'Actual: {y_test_actual[-1][0]:.2f}', ha='left', va='center', fontsize=10)

    plt.text(date_range[-1] + pd.DateOffset(days=3), y_pred_actual[-1][0] - 2,
             f'Predicted: {y_pred_actual[-1][0]:.2f}', ha='right', va='center', fontsize=10)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{symbol} Stock Price Prediction")
    plt.legend()
    plt.show()

def run(symbol, start_date="2020-01-01", end_date="2023-01-01", sequence_length=10, epochs=100, batch_size=32):
    stock_data = fetch_data(symbol, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, sequence_length)
    model = build_model(sequence_length)
    progress = ttk.Progressbar(root, length=200, mode='indeterminate')
    progress.grid(row=4, column=0, pady=10, sticky="ew")
    progress.start()

    def on_progress_done():
        # Stop the progress bar
        progress.stop()
        progress.grid_forget()

    # Run the model training in a separate thread
    import threading

    def train_model_thread():
        try:
            train_model(model, X_train, y_train, epochs, batch_size)
            y_pred = predict_data(model, X_test)
            y_test_actual = inverse_transform(scaler, y_test)
            y_pred_actual = inverse_transform(scaler, y_pred)
            date_range = stock_data.index[-len(y_test):]
            plot_predictions(stock_data, date_range, y_test_actual, y_pred_actual, symbol)
        finally:
            # Hide the progress bar after processing is done
            root.after(1, on_progress_done)

    threading.Thread(target=train_model_thread).start()

# usage:

   

def run_application():
    
    # Create a new window for the application
    app_window = tk.Toplevel(root)
    app_window.title("Application Page")

    # Configure ttk style for a modern theme
    style = ttk.Style()
    style.configure("TButton", padding=10, font=("Helvetica", 12))

    # Create and place ttk widgets in the new window
    frame = ttk.Frame(app_window, padding="10")
    frame.grid(row=0, column=0, sticky="nsew")

    search_label = ttk.Label(frame, text="Search:")
    search_entry = ttk.Entry(frame, width=30)
    search_button = ttk.Button(frame, text="Search", command=lambda: search_function(search_entry.get()))
    exit_app_button = ttk.Button(frame, text="Exit", command=app_window.destroy)

    # Arrange widgets in a grid layout
    search_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
    search_entry.grid(row=0, column=1, padx=10, pady=10)
    search_button.grid(row=0, column=2, padx=10, pady=10)
    exit_app_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    # Configure row and column weights to allow resizing
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(0, weight=1)

def search_function(query):
    result_label.config(text=f"Search query: {query}")
    run(query)
    result_label.config(text=f"Stock prediction for {query} completed.")
def browse_file():
    file_path = filedialog.askopenfilename(title="Select a file")
    result_label.config(text=f"Selected file: {file_path}")

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Stock Prediction App")

# Configure ttk style for a modern theme
style = ttk.Style()
style.configure("TButton", padding=10, font=("Helvetica", 12))

# Create ttk buttons
run_button = ttk.Button(root, text="Run Application", command=run_application)
browse_button = ttk.Button(root, text="Browse", command=browse_file)
exit_button = ttk.Button(root, text="Exit", command=exit_app)

# Display buttons in a grid layout, aligning them to the top left
run_button.grid(row=0, column=0, sticky="w", padx=0, pady=0)
browse_button.grid(row=1, column=0, sticky="w", padx=0, pady=0)
exit_button.grid(row=2, column=0, sticky="w", padx=0, pady=0)

# Display a label to show results or messages
result_label = ttk.Label(root, text="", font=("Helvetica", 12))
result_label.grid(row=3, column=0, pady=10, sticky="ew")

# Configure row and column weights to allow resizing
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)

# Configure resizing behavior
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)


# Run the Tkinter event loop
root.mainloop()
