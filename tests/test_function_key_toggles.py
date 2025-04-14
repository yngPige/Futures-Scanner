#!/usr/bin/env python
"""
Test script for function key toggles in the Terminal UI.

This script runs the Terminal UI with function key toggles at the bottom of the screen.
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

from src.ui.terminal_ui import TerminalUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    logger.info("Testing Terminal UI with function key toggles")
    
    # Initialize the Terminal UI
    ui = TerminalUI()
    
    # Set initial toggle states
    ui.settings['use_gpu'] = False
    ui.settings['save'] = True
    ui.settings['tune'] = False
    ui.settings['use_llm'] = True
    
    # Print instructions
    print("\n" + "=" * 80)
    print("Function Key Toggle Test")
    print("=" * 80)
    print("This test demonstrates the function key toggles at the bottom of the screen.")
    print("You can toggle the options using the following function keys:")
    print("  [F1] - Toggle GPU Acceleration")
    print("  [F2] - Toggle Save Results")
    print("  [F3] - Toggle Hyperparameter Tuning")
    print("  [F4] - Toggle LLM Analysis")
    print("When an option is enabled, it will be displayed in green text.")
    print("\nNote: You may need to install the keyboard module with:")
    print("  pip install keyboard")
    print("\nPress Enter to continue...")
    input()
    
    # Run the Terminal UI
    ui.run()

if __name__ == "__main__":
    main()
