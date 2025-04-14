"""
Test script for fetching symbols using the CCXT:ALL exchange implementation.

This script tests fetching available symbols from multiple exchanges using the CCXT:ALL option.
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
    print("Testing CCXT:ALL Symbol Fetching")
    print("==============================")
    
    # Create a DataFetcher with CCXT:ALL
    print("\nCreating DataFetcher with CCXT:ALL...")
    fetcher = DataFetcher(exchange_id='CCXT:ALL', timeframe='1h')
    
    # Limit to just a few exchanges for testing
    test_exchanges = ['kraken', 'kucoin']
    fetcher.all_exchanges = test_exchanges
    print(f"Limited to exchanges: {fetcher.all_exchanges}")
    
    # Get available symbols
    print("\nFetching available USDT symbols...")
    symbols = fetcher.get_available_symbols(quote_currency='USDT')
    
    print(f"Found {len(symbols)} unique symbols across exchanges")
    
    # Show the first 20 symbols
    print("\nFirst 20 symbols:")
    for i, symbol in enumerate(symbols[:20], 1):
        print(f"{i}. {symbol}")
    
    # Try to fetch data for a few of these symbols
    if symbols:
        test_symbols = symbols[:3]  # Test the first 3 symbols
        print(f"\nTesting data fetching for symbols: {test_symbols}")
        
        for symbol in test_symbols:
            print(f"\nFetching data for {symbol}...")
            df = fetcher.fetch_ohlcv(symbol, limit=5)
            
            if df is not None and not df.empty:
                print(f"Successfully fetched {len(df)} rows of data from {df.attrs.get('exchange', 'Unknown')}")
                print(f"Latest price: {df['close'].iloc[-1]:.2f}")
            else:
                print(f"Failed to fetch data for {symbol}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
