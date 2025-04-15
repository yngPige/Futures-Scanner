#!/usr/bin/env python
"""
Test script for the Tkinter Symbol Selector with enhanced notification system.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the symbol selector
from src.ui.tkinter_symbol_selector import select_symbol_tkinter

def main():
    """Test the symbol selector with enhanced notification system."""
    print("Testing Tkinter Symbol Selector with enhanced notification system...")
    
    # Run the symbol selector
    symbol, exchange, cancelled = select_symbol_tkinter()
    
    # Print the results
    if cancelled:
        print("Selection cancelled")
    else:
        print(f"Selected symbol: {symbol} on {exchange}")

if __name__ == "__main__":
    main()
