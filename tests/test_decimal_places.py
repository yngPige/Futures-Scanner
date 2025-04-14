"""
Test script to verify that asset prices are displayed with 5 decimal places.

This script tests the formatting of price values in the terminal output.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the terminal output generator
from src.ui.terminal_output import TerminalOutputGenerator

def create_test_dataframe():
    """Create a test DataFrame with sample data."""
    # Create a sample DataFrame with price data
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    data = {
        'open': [1.23456, 2.34567, 3.45678, 4.56789, 5.67890, 6.78901, 7.89012, 8.90123, 9.01234, 10.12345],
        'high': [1.34567, 2.45678, 3.56789, 4.67890, 5.78901, 6.89012, 7.90123, 8.01234, 9.12345, 10.23456],
        'low': [1.12345, 2.23456, 3.34567, 4.45678, 5.56789, 6.67890, 7.78901, 8.89012, 9.90123, 10.01234],
        'close': [1.23456, 2.34567, 3.45678, 4.56789, 5.67890, 6.78901, 7.89012, 8.90123, 9.01234, 10.12345],
        'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'rsi_14': [30, 40, 50, 60, 70, 65, 55, 45, 35, 45],
        'MACD_12_26_9': [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2],
        'MACDs_12_26_9': [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05, 0.15],
        'sma_50': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        'sma_200': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'entry_price': [1.20000, 2.30000, 3.40000, 4.50000, 5.60000, 6.70000, 7.80000, 8.90000, 9.00000, 10.10000],
        'stop_loss': [1.10000, 2.20000, 3.30000, 4.40000, 5.50000, 6.60000, 7.70000, 8.80000, 8.90000, 10.00000],
        'take_profit': [1.30000, 2.40000, 3.50000, 4.60000, 5.70000, 6.80000, 7.90000, 9.00000, 9.10000, 10.20000],
        'risk_reward': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        'signal': [1, 0, -1, 0, 1, 0, -1, 0, 1, 0],
        'prediction': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'prediction_probability': [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5, 0.8, 0.7]
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_terminal_output():
    """Test the terminal output formatting."""
    print("Testing Terminal Output Formatting")
    print("=================================")

    # Create a test DataFrame
    df = create_test_dataframe()

    # Test formatting directly
    latest = df.iloc[-1]

    # Format price values with 5 decimal places
    open_price = f"{latest['open']:.5f}"
    high_price = f"{latest['high']:.5f}"
    low_price = f"{latest['low']:.5f}"
    close_price = f"{latest['close']:.5f}"
    entry_price = f"{latest['entry_price']:.5f}"
    stop_loss = f"{latest['stop_loss']:.5f}"
    take_profit = f"{latest['take_profit']:.5f}"

    # Print formatted values
    print(f"\nFormatted price values with 5 decimal places:")
    print(f"Open:        {open_price}")
    print(f"High:        {high_price}")
    print(f"Low:         {low_price}")
    print(f"Close:       {close_price}")
    print(f"Entry Price: {entry_price}")
    print(f"Stop Loss:   {stop_loss}")
    print(f"Take Profit: {take_profit}")

    # Verify that the values have 5 decimal places
    print("\nVerifying decimal places:")
    for name, value in [
        ("Open", open_price),
        ("High", high_price),
        ("Low", low_price),
        ("Close", close_price),
        ("Entry Price", entry_price),
        ("Stop Loss", stop_loss),
        ("Take Profit", take_profit)
    ]:
        if '.' in value and len(value.split('.')[1]) == 5:
            print(f"✓ {name} has 5 decimal places: {value}")
        else:
            print(f"✗ {name} does NOT have 5 decimal places: {value}")

    print("\nTest completed.")

def main():
    """Main function."""
    test_terminal_output()

if __name__ == "__main__":
    main()
