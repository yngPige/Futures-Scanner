"""
Terminal UI Entry Point for Crypto Futures Scanner

This file applies the numpy monkey patch and then launches the terminal UI.
"""

# Apply patches first
import sys
import os

# Try to patch typing_extensions
try:
    from scripts.build.hooks import typing_extensions_patch
    typing_extensions_patch.patch_typing_extensions()
except ImportError:
    print("Warning: typing_extensions_patch not found")

# Import the numpy monkey patch
from src.utils import monkey_patch

# Set up error handling
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions."""
    # Log the error
    import traceback
    print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Keep the console window open
    input("\nPress Enter to exit...")
    return sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set the exception hook
sys.excepthook = handle_exception

# Import the terminal UI
try:
    from src.ui.terminal_ui import main

    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
