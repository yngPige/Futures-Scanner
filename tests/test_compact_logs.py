"""
Test script to demonstrate the compact log format.

This script simulates a prediction process and displays the logs in the new compact format.
"""

import time
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the terminal UI
from src.ui.terminal_ui import TerminalUI

def main():
    """Main function."""
    # Initialize terminal UI
    ui = TerminalUI()

    # Clear screen and show header
    ui.clear_screen()
    ui.print_header("3lacks Scanner - Test Compact Log Format")

    # Simulate a series of operations with compact logs
    ui.print_info("Starting compact log test...")
    
    # Simulate data fetching
    ui.print_info("Fetching data for BTC/USDT from kraken...")
    ui.show_loading_animation("Retrieving market data", duration=1, compact_completion=True)
    
    # Simulate technical analysis
    ui.print_info("Performing technical analysis...")
    ui.show_loading_animation("Calculating technical indicators", duration=1, compact_completion=True)
    
    # Simulate model training
    ui.print_info("Training Random Forest model...")
    ui.show_loading_animation("Training machine learning model", duration=1, compact_completion=True)
    
    # Wait for user input
    print("\nTest completed. The logs should now be displayed in a more compact format.")
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()
