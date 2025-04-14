"""
Test script for CCXT caching and filtering complex pairs.

This script tests the caching functionality and filtering of complex trading pairs.
"""

import sys
import os
import logging
import re
import time

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the DataFetcher
from src.data.data_fetcher import DataFetcher, COMPLEX_PAIR_PATTERN

def test_complex_pair_pattern():
    """Test the complex pair pattern matching."""
    print("\nTesting complex pair pattern matching:")
    
    # Test pairs that should match (complex pairs to be filtered out)
    complex_pairs = [
        "BTC/USDT:USDT-250415-76000-C",
        "BTC/USDT:USDT-250415-76000-P",
        "ETH/USDT:USDT-250415-77000-C",
        "ETH/USDT:USDT-250415-77000-P",
        "BTC/USDT:USDT-250415-78000-C",
        "BTC/USDT:USDT-250415-78000-P",
        "ETH/USDT:USDT-250415-79000-C",
        "ETH/USDT:USDT-250415-79000-P",
        "BTC/USDT:USDT-250415-80000-C",
        "BTC/USDT:USDT-250415-80000-P",
        "ETH/USDT:USDT-250415-81000-C",
        "ETH/USDT:USDT-250415-81000-P"
    ]
    
    # Test pairs that should not match (normal pairs to keep)
    normal_pairs = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "DOGE/USDT"
    ]
    
    # Test the pattern
    print("Testing complex pairs (should match and be filtered out):")
    for pair in complex_pairs:
        if COMPLEX_PAIR_PATTERN.match(pair):
            print(f"✓ Correctly matched: {pair}")
        else:
            print(f"✗ Failed to match: {pair}")
    
    print("\nTesting normal pairs (should not match and be kept):")
    for pair in normal_pairs:
        if not COMPLEX_PAIR_PATTERN.match(pair):
            print(f"✓ Correctly did not match: {pair}")
        else:
            print(f"✗ Incorrectly matched: {pair}")

def test_caching():
    """Test the caching functionality."""
    print("\nTesting caching functionality:")
    
    # Create a DataFetcher
    exchange_id = 'kraken'  # Use a reliable exchange for testing
    fetcher = DataFetcher(exchange_id=exchange_id, timeframe='1h')
    
    # Get the cache path
    cache_path = fetcher._get_cache_path(exchange_id, 'spot', 'USDT')
    print(f"Cache path: {cache_path}")
    
    # Delete the cache file if it exists
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Deleted existing cache file: {cache_path}")
    
    # First fetch - should create cache
    print("\nFirst fetch (should create cache):")
    start_time = time.time()
    symbols = fetcher.get_available_symbols(quote_currency='USDT')
    end_time = time.time()
    print(f"Fetched {len(symbols)} symbols in {end_time - start_time:.2f} seconds")
    
    # Check if cache file was created
    if os.path.exists(cache_path):
        print(f"✓ Cache file was created: {cache_path}")
    else:
        print(f"✗ Cache file was not created: {cache_path}")
    
    # Second fetch - should use cache
    print("\nSecond fetch (should use cache):")
    start_time = time.time()
    cached_symbols = fetcher.get_available_symbols(quote_currency='USDT')
    end_time = time.time()
    print(f"Fetched {len(cached_symbols)} symbols in {end_time - start_time:.2f} seconds")
    
    # Check if the second fetch was faster
    if end_time - start_time < 0.1:  # Should be very fast if using cache
        print("✓ Second fetch was much faster (used cache)")
    else:
        print("✗ Second fetch was not faster (did not use cache)")
    
    # Check if the results are the same
    if set(symbols) == set(cached_symbols):
        print("✓ Results from cache match original results")
    else:
        print("✗ Results from cache do not match original results")

def test_filtering():
    """Test the filtering of complex pairs."""
    print("\nTesting filtering of complex pairs:")
    
    # Create a list with both normal and complex pairs
    mixed_pairs = [
        "BTC/USDT",
        "ETH/USDT",
        "BTC/USDT:USDT-250415-76000-C",
        "BTC/USDT:USDT-250415-76000-P",
        "SOL/USDT",
        "ADA/USDT",
        "ETH/USDT:USDT-250415-77000-C",
        "ETH/USDT:USDT-250415-77000-P",
        "XRP/USDT",
        "DOGE/USDT"
    ]
    
    # Create a DataFetcher
    fetcher = DataFetcher(exchange_id='kraken', timeframe='1h')
    
    # Filter the pairs
    filtered_pairs = fetcher._filter_complex_pairs(mixed_pairs)
    
    # Check the results
    print(f"Original pairs: {len(mixed_pairs)}")
    print(f"Filtered pairs: {len(filtered_pairs)}")
    
    # Check which pairs were filtered out
    filtered_out = set(mixed_pairs) - set(filtered_pairs)
    print("\nPairs that were filtered out:")
    for pair in filtered_out:
        print(f"- {pair}")
    
    # Check if all complex pairs were filtered out
    complex_count = sum(1 for pair in mixed_pairs if COMPLEX_PAIR_PATTERN.match(pair))
    if len(filtered_out) == complex_count:
        print(f"\n✓ All {complex_count} complex pairs were filtered out")
    else:
        print(f"\n✗ Only {len(filtered_out)} out of {complex_count} complex pairs were filtered out")

def main():
    """Main function."""
    print("Testing CCXT Caching and Filtering")
    print("=================================")
    
    # Test the complex pair pattern
    test_complex_pair_pattern()
    
    # Test the caching functionality
    test_caching()
    
    # Test the filtering of complex pairs
    test_filtering()
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
