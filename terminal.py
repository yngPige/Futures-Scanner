"""
Terminal UI Entry Point for Crypto Futures Scanner

This file applies the numpy monkey patch and then launches the terminal UI.
"""

# Import the monkey patch first
import monkey_patch

# Import the terminal UI
from src.ui.terminal_ui import main

if __name__ == "__main__":
    main()
