#!/usr/bin/env python
"""
Test script for the caching functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the helpers module
from src.utils.helpers import (
    save_settings, load_settings, 
    save_analysis_to_cache, load_analysis_from_cache,
    get_cache_directory
)

def test_settings():
    """Test the settings functionality."""
    print("\nTesting settings functionality:")
    
    # Create test settings
    test_settings = {
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'limit': 500,
        'exchange': 'kraken',
        'model_type': 'random_forest',
        'model_path': None,
        'save': True,
        'tune': False,
        'use_gpu': True
    }
    
    # Save settings
    print("Saving settings...")
    save_settings(test_settings)
    
    # Load settings
    print("Loading settings...")
    loaded_settings = load_settings()
    
    # Check if settings were loaded correctly
    if loaded_settings:
        print("Settings loaded successfully!")
        print(f"Symbol: {loaded_settings.get('symbol')}")
        print(f"Exchange: {loaded_settings.get('exchange')}")
        print(f"Timeframe: {loaded_settings.get('timeframe')}")
    else:
        print("Failed to load settings!")
    
    return loaded_settings is not None

def test_analysis_cache():
    """Test the analysis cache functionality."""
    print("\nTesting analysis cache functionality:")
    
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'close': np.random.normal(100, 5, 100),
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'volume': np.random.normal(1000, 100, 100),
        'rsi_14': np.random.normal(50, 10, 100),
        'sma_50': np.random.normal(100, 3, 100),
        'sma_200': np.random.normal(98, 3, 100),
        'prediction': np.random.choice([0, 1], 100),
        'target': np.random.choice([0, 1], 100)
    }, index=dates)
    
    # Save analysis to cache
    symbol = 'BTC/USDT'
    exchange = 'kraken'
    timeframe = '1h'
    
    print(f"Saving analysis for {symbol} on {exchange} ({timeframe}) to cache...")
    save_analysis_to_cache(df, symbol, exchange, timeframe)
    
    # Load analysis from cache
    print(f"Loading analysis for {symbol} on {exchange} ({timeframe}) from cache...")
    cached_df = load_analysis_from_cache(symbol, exchange, timeframe)
    
    # Check if analysis was loaded correctly
    if cached_df is not None:
        print("Analysis loaded successfully!")
        print(f"DataFrame shape: {cached_df.shape}")
        print(f"Columns: {', '.join(cached_df.columns[:5])}...")
    else:
        print("Failed to load analysis from cache!")
    
    return cached_df is not None

def main():
    """Main function to run the tests."""
    print("=" * 50)
    print("Testing Caching Functionality")
    print("=" * 50)
    
    # Print cache directory
    cache_dir = get_cache_directory()
    print(f"Cache directory: {cache_dir}")
    
    # Test settings
    settings_success = test_settings()
    
    # Test analysis cache
    cache_success = test_analysis_cache()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Settings functionality: {'✓ Passed' if settings_success else '✗ Failed'}")
    print(f"Analysis cache functionality: {'✓ Passed' if cache_success else '✗ Failed'}")
    
    # Wait for user input
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
