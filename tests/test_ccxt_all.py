"""
Test script for the CCXT:ALL exchange implementation.

This script tests the ability to fetch data from multiple exchanges using the CCXT:ALL option.
"""

import sys
import os
import logging
from datetime import datetime

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the DataFetcher
from src.data.data_fetcher import DataFetcher

def test_available_exchanges():
    """Test getting available exchanges."""
    print("\nTesting available exchanges:")

    # Get available exchanges directly
    fetcher = DataFetcher(exchange_id='kraken', timeframe='1h')  # Use any exchange here
    exchanges = fetcher.get_available_exchanges()

    print(f"Found {len(exchanges)} available exchanges")
    print(f"First 10 exchanges: {exchanges[:10]}")

    return exchanges

def test_available_symbols():
    """Test getting available symbols from multiple exchanges."""
    print("\nTesting available symbols:")

    # Create a DataFetcher with CCXT:ALL but limit to just 2 exchanges for testing
    fetcher = DataFetcher(exchange_id='CCXT:ALL', timeframe='1h')
    # Limit the exchanges to test with
    fetcher.all_exchanges = fetcher.all_exchanges[:2]  # Just use the first 2 exchanges
    print(f"Testing with exchanges: {fetcher.all_exchanges}")

    # Get available symbols
    symbols = fetcher.get_available_symbols(quote_currency='USDT')

    print(f"Found {len(symbols)} unique symbols across exchanges")
    print(f"First 10 symbols: {symbols[:10]}")

    return symbols

def test_fetch_data():
    """Test fetching data from multiple exchanges."""
    print("\nTesting data fetching:")

    # Create a DataFetcher with CCXT:ALL but limit to just 2 exchanges for testing
    fetcher = DataFetcher(exchange_id='CCXT:ALL', timeframe='1h')
    # Limit the exchanges to test with
    fetcher.all_exchanges = ['kraken', 'kucoin']  # Use specific exchanges that are likely to work
    print(f"Testing with exchanges: {fetcher.all_exchanges}")

    # Try to fetch data for BTC/USDT
    print("Fetching BTC/USDT data...")
    df = fetcher.fetch_ohlcv('BTC/USDT', limit=100)

    if df is not None and not df.empty:
        print(f"Successfully fetched {len(df)} rows of data")
        print(f"Data from exchange: {df.attrs.get('exchange', 'Unknown')}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Latest price: {df['close'].iloc[-1]:.2f}")

        # Show a sample of the data
        print("\nSample data:")
        print(df.head(3))
    else:
        print("Failed to fetch data")

    return df

def main():
    """Main function."""
    print("Testing CCXT:ALL Exchange Implementation")
    print("=======================================")

    # Test getting available exchanges
    exchanges = test_available_exchanges()

    # Test getting available symbols
    symbols = test_available_symbols()

    # Test fetching data
    df = test_fetch_data()

    print("\nTest completed.")

if __name__ == "__main__":
    main()
