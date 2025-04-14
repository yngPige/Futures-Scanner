"""
Simple test script for the CCXT:ALL exchange implementation.

This script tests the basic functionality of the CCXT:ALL option.
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
    print("Testing CCXT:ALL Exchange Implementation (Simple Test)")
    print("===================================================")
    
    # Create a DataFetcher with CCXT:ALL
    print("\nCreating DataFetcher with CCXT:ALL...")
    fetcher = DataFetcher(exchange_id='CCXT:ALL', timeframe='1h')
    
    # Check if all_exchanges is populated
    print(f"Number of available exchanges: {len(fetcher.all_exchanges)}")
    print(f"First 5 exchanges: {fetcher.all_exchanges[:5]}")
    
    # Try to fetch data for BTC/USDT from a specific exchange
    print("\nFetching BTC/USDT data from kraken...")
    kraken_fetcher = DataFetcher(exchange_id='kraken', timeframe='1h')
    df = kraken_fetcher.fetch_ohlcv('BTC/USDT', limit=10)
    
    if df is not None and not df.empty:
        print(f"Successfully fetched {len(df)} rows of data from kraken")
        print(f"Data attributes: {df.attrs}")
        print(f"Latest price: {df['close'].iloc[-1]:.2f}")
    else:
        print("Failed to fetch data from kraken")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
