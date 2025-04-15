#!/usr/bin/env python
"""
Test script for the Tkinter Symbol Selector with auto-focus functionality.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the symbol selector
from src.ui.tkinter_symbol_selector import select_symbol_tkinter

def main():
    """Test the symbol selector with auto-focus functionality."""
    print("Testing Tkinter Symbol Selector with auto-focus functionality...")
    print("The search box should be automatically focused when the window opens.")
    print("You should be able to start typing immediately to search for symbols.")
    
    # Run the symbol selector
    symbol, exchange, cancelled = select_symbol_tkinter()
    
    # Print the results
    if cancelled:
        print("Selection cancelled")
    else:
        print(f"Selected symbol: {symbol} on {exchange}")

if __name__ == "__main__":
    main()
