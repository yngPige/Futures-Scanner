"""
Test script for fetching data using the CCXT:ALL exchange implementation.

This script tests fetching data from multiple exchanges using the CCXT:ALL option.
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
    print("Testing CCXT:ALL Data Fetching")
    print("=============================")
    
    # Create a DataFetcher with CCXT:ALL
    print("\nCreating DataFetcher with CCXT:ALL...")
    fetcher = DataFetcher(exchange_id='CCXT:ALL', timeframe='1h')
    
    # Limit to just a few exchanges for testing
    test_exchanges = ['kraken', 'kucoin', 'huobi']
    fetcher.all_exchanges = test_exchanges
    print(f"Limited to exchanges: {fetcher.all_exchanges}")
    
    # Try to fetch data for BTC/USDT
    print("\nFetching BTC/USDT data...")
    df = fetcher.fetch_ohlcv('BTC/USDT', limit=10)
    
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
    
    # Try to fetch data for ETH/USDT
    print("\nFetching ETH/USDT data...")
    df = fetcher.fetch_ohlcv('ETH/USDT', limit=10)
    
    if df is not None and not df.empty:
        print(f"Successfully fetched {len(df)} rows of data")
        print(f"Data from exchange: {df.attrs.get('exchange', 'Unknown')}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Latest price: {df['close'].iloc[-1]:.2f}")
    else:
        print("Failed to fetch data")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
