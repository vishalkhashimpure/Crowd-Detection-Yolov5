import tkinter as tk
from tkinter import Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_graph(minute_counts, graph_type="line"):
    """
    Display a line or bar graph for minute counts in a new Tkinter window, with a text summary.

    Args:
        minute_counts (dict): A dictionary with minutes as keys and counts as values.
        graph_type (str): Type of graph to display, either 'line' or 'bar'. Defaults to 'line'.
    """
    # Create a new top-level Tkinter window
    graph_window = Toplevel()
    graph_window.title("Minute Counts Graph")
    graph_window.geometry("900x700")

    # Create a frame for the text summary
    text_frame = tk.Frame(graph_window)
    text_frame.pack(fill="x", pady=10)

    # Display the text summary
    lbl_summary = tk.Label(text_frame, text="Minute Counts Summary:", font=("Arial", 12), anchor="w")
    lbl_summary.pack(anchor="w", padx=10)

    text_box = tk.Text(text_frame, height=8, font=("Arial", 10), wrap="word")
    text_box.pack(fill="x", padx=10, pady=5)

    # Format and insert the counts into the text box
    counts_summary = "\n".join([f"{minute}: {count} people" for minute, count in sorted(minute_counts.items())])
    text_box.insert("1.0", counts_summary)
    text_box.config(state="disabled")  # Make the text box read-only

    # Create the figure for the graph
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_minutes = sorted(minute_counts.keys())
    counts = [minute_counts[minute] for minute in sorted_minutes]

    # Plot the graph based on the selected type
    if graph_type == "line":
        ax.plot(sorted_minutes, counts, marker='o', label="People Count")
    elif graph_type == "bar":
        ax.bar(sorted_minutes, counts, label="People Count", color='blue', alpha=0.7)

    # Configure the graph
    ax.set_title("People Count Per Minute")
    ax.set_xlabel("Time (Minute)")
    ax.set_ylabel("People Count")
    ax.set_xticks(range(len(sorted_minutes)))
    ax.set_xticklabels(sorted_minutes, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True)

    # Embed the graph into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Add a close button
    btn_close = tk.Button(graph_window, text="Close", command=graph_window.destroy)
    btn_close.pack(pady=10)
