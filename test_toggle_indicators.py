"""
Test script for toggle indicators in the Terminal UI.

This script runs the Terminal UI with toggle indicators at the bottom of the screen.
"""

import os
import sys
import logging
from datetime import datetime

# Import the monkey patch first
import monkey_patch

from src.ui.terminal_ui import TerminalUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    logger.info("Testing Terminal UI with toggle indicators")
    
    # Initialize the Terminal UI
    ui = TerminalUI()
    
    # Set initial toggle states
    ui.settings['use_gpu'] = False
    ui.settings['save'] = True
    ui.settings['tune'] = False
    ui.settings['use_llm'] = True
    
    # Run the Terminal UI
    ui.run()

if __name__ == "__main__":
    main()
