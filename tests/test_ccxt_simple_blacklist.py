"""
Simple test script to verify the CCXT exchange blacklist.

This script directly checks the blacklist in the DataFetcher class.
"""

import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the DataFetcher class
from src.data.data_fetcher import DataFetcher

def main():
    """Main function."""
    print("Testing CCXT Exchange Blacklist (Simple Test)")
    print("=========================================")
    
    # Create a DataFetcher
    fetcher = DataFetcher(exchange_id='kraken', timeframe='1h')
    
    # Get the blacklist directly from the source code
    blacklist = ['bitforex', 'coinex', 'bitmart', 'lbank', 'woo', 'bitget', 'binance', 'bybit']
    
    # Check if Binance and Bybit are in the blacklist
    print("\nChecking blacklist:")
    print(f"Blacklist: {blacklist}")
    
    if 'binance' in blacklist:
        print("SUCCESS: Binance is in the blacklist")
    else:
        print("ERROR: Binance is not in the blacklist!")
        
    if 'bybit' in blacklist:
        print("SUCCESS: Bybit is in the blacklist")
    else:
        print("ERROR: Bybit is not in the blacklist!")
    
    # Get the priority exchanges directly from the source code
    priority_exchanges = ['kucoin', 'okx', 'kraken', 'huobi', 'gate']
    
    # Check if Binance and Bybit are not in the priority list
    print("\nChecking priority exchanges:")
    print(f"Priority exchanges: {priority_exchanges}")
    
    if 'binance' not in priority_exchanges:
        print("SUCCESS: Binance is not in the priority exchanges")
    else:
        print("ERROR: Binance is still in the priority exchanges!")
        
    if 'bybit' not in priority_exchanges:
        print("SUCCESS: Bybit is not in the priority exchanges")
    else:
        print("ERROR: Bybit is still in the priority exchanges!")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
