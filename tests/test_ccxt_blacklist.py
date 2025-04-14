"""
Test script to verify that Binance and Bybit are excluded from CCXT:ALL.

This script checks that Binance and Bybit are properly blacklisted from the CCXT:ALL exchange option.
"""

import sys
import os
import logging

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the DataFetcher
from src.data.data_fetcher import DataFetcher

def main():
    """Main function."""
    print("Testing CCXT Exchange Blacklist")
    print("==============================")

    # Create a DataFetcher and call get_available_exchanges directly
    print("\nGetting available exchanges...")
    fetcher = DataFetcher(exchange_id='kraken', timeframe='1h')  # Use any exchange here
    exchanges = fetcher.get_available_exchanges()

    print(f"Found {len(exchanges)} available exchanges")

    # Check if Binance and Bybit are excluded
    if 'binance' in exchanges:
        print("ERROR: Binance is still in the exchange list!")
    else:
        print("SUCCESS: Binance is properly excluded from the exchange list")

    if 'bybit' in exchanges:
        print("ERROR: Bybit is still in the exchange list!")
    else:
        print("SUCCESS: Bybit is properly excluded from the exchange list")

    # Show the first 10 exchanges
    print("\nFirst 10 exchanges:")
    for i, exchange in enumerate(exchanges[:10], 1):
        print(f"{i}. {exchange}")

    print("\nTest completed.")

if __name__ == "__main__":
    main()
