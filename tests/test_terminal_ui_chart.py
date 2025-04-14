"""
Test script for the Terminal UI with terminal charts.

This script runs the Terminal UI with the terminal chart option.
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
    logger.info("Testing Terminal UI with terminal charts")
    
    # Initialize the Terminal UI
    ui = TerminalUI()
    
    # Run the Terminal UI
    ui.run()

if __name__ == "__main__":
    main()
