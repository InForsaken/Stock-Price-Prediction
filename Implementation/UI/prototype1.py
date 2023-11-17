import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

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
    # Implement your search logic here
    result_label.config(text=f"Search query: {query}")

def browse_file():
    file_path = filedialog.askopenfilename(title="Select a file")
    result_label.config(text=f"Selected file: {file_path}")

def exit_app():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Button UI Example")

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