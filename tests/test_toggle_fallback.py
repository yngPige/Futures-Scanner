#!/usr/bin/env python
"""
Test script for toggle fallback mechanism in the Terminal UI.

This script tests the fallback mechanism for toggle keys when the keyboard module is not available.
"""

import os
import sys
import logging
from datetime import datetime

# Import the monkey patch first
import sys
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the monkey patch
from src.utils import monkey_patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    logger.info("Testing Terminal UI with toggle fallback mechanism")
    
    # Simulate keyboard module not being available
    import src.ui.terminal_ui
    src.ui.terminal_ui.KEYBOARD_AVAILABLE = False
    
    # Import the TerminalUI class after setting KEYBOARD_AVAILABLE to False
    from src.ui.terminal_ui import TerminalUI
    
    # Initialize the Terminal UI
    ui = TerminalUI()
    
    # Set initial toggle states
    ui.settings['use_gpu'] = False
    ui.settings['save'] = True
    ui.settings['tune'] = False
    ui.settings['use_llm'] = True
    
    # Print instructions
    print("\n" + "=" * 80)
    print("Toggle Fallback Test")
    print("=" * 80)
    print("This test demonstrates the toggle fallback mechanism when the keyboard module is not available.")
    print("You can toggle the options using the following letter keys:")
    print("  [g] - Toggle GPU Acceleration")
    print("  [s] - Toggle Save Results")
    print("  [t] - Toggle Hyperparameter Tuning")
    print("  [l] - Toggle LLM Analysis")
    print("When an option is enabled, it will be displayed in green text.")
    print("\nPress Enter to continue...")
    input()
    
    # Run the Terminal UI
    ui.run()

if __name__ == "__main__":
    main()
