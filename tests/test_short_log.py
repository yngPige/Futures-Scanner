"""
Test script to demonstrate the shortened log format.

This script simulates a prediction process and displays the logs in the new concise format.
"""

import time
import random
import logging
from datetime import datetime

# Import the monkey patch first
import sys
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the monkey patch
from src.utils import monkey_patch

from src.ui.terminal_ui import TerminalUI

# Configure logging using custom logging utility
from src.utils.logging_utils import configure_logging

# Configure logging to only show errors in console and save to error log file
configure_logging()
logger = logging.getLogger(__name__)

def simulate_prediction_process(ui):
    """Simulate a prediction process with various log messages."""
    # Define the steps in the prediction process
    data_steps = [
        "Authenticating connection...",
        "Setting up data parameters...",
        "Requesting historical data...",
        "Downloading candles...",
        "Processing OHLCV data...",
        "Validating timestamps...",
        "Formatting data structures...",
        "Checking for missing values...",
        "Finalizing data retrieval..."
    ]

    analysis_steps = [
        "Analyzing market patterns...",
        "Processing technical indicators...",
        "Calculating momentum signals...",
        "Evaluating trend strength...",
        "Checking support/resistance levels...",
        "Validating data integrity...",
        "Optimizing analysis parameters...",
        "Applying machine learning models...",
        "Finalizing results..."
    ]

    prediction_steps = [
        "Preparing feature data...",
        "Normalizing inputs...",
        "Running model inference...",
        "Calculating prediction probabilities...",
        "Generating trading signals...",
        "Evaluating prediction confidence...",
        "Applying risk filters...",
        "Formatting prediction results...",
        "Finalizing predictions..."
    ]

    # Initialize collected_logs if it doesn't exist
    if not hasattr(ui, 'collected_logs'):
        ui.collected_logs = []

    # Simulate data retrieval
    ui.print_info("Starting data retrieval process...")
    for step in data_steps:
        # Add log to collected_logs with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ui.collected_logs.append((timestamp, step))
        time.sleep(0.1)  # Simulate processing time

    # Add completion log
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ui.collected_logs.append((timestamp, "✓ Retrieving market data completed!"))

    # Simulate analysis
    ui.print_info("Starting analysis process...")
    for step in analysis_steps:
        # Add log to collected_logs with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ui.collected_logs.append((timestamp, step))
        time.sleep(0.1)  # Simulate processing time

    # Add completion log
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ui.collected_logs.append((timestamp, "✓ Calculating technical indicators completed!"))

    # Simulate prediction
    ui.print_info("Starting prediction process...")
    for step in prediction_steps:
        # Add log to collected_logs with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ui.collected_logs.append((timestamp, step))
        time.sleep(0.1)  # Simulate processing time

    # Add completion log
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ui.collected_logs.append((timestamp, "✓ Running prediction model completed!"))

    # Simulate final step
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ui.collected_logs.append((timestamp, "✓ Preparing prediction report completed!"))

def main():
    """Main function."""
    # Initialize terminal UI
    ui = TerminalUI()

    # Clear screen and show header
    ui.clear_screen()
    ui.print_header("3lacks Scanner - Test Short Log Format")

    # Simulate prediction process
    ui.print_info("Starting prediction process simulation...")
    simulate_prediction_process(ui)

    # Display logs in concise format
    ui.display_collected_logs(title="Prediction Process Log - BTC/USDT")

    # Wait for user input
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
