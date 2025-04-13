"""
Test script for the main application with terminal charts.

This script runs the main application with the terminal chart option.
"""

import argparse
import logging
import sys
from datetime import datetime

# Import the monkey patch first
import monkey_patch

from main import fetch_data, analyze_data, visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Args:
    """Mock arguments class."""
    def __init__(self):
        self.exchange = 'coinbase'
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
        self.limit = 100
        self.theme = 'dark'
        self.terminal_chart = True
        self.interactive = False
        self.save = False
        self.no_display = False

def main():
    """Main function."""
    logger.info("Testing main application with terminal charts")
    
    # Create mock arguments
    args = Args()
    
    # Fetch data
    logger.info(f"Fetching data for {args.symbol} from {args.exchange}")
    df = fetch_data(args)
    
    if df is None or df.empty:
        logger.error(f"Failed to fetch data for {args.symbol}")
        return
    
    logger.info(f"Successfully fetched {len(df)} candles")
    
    # Analyze data
    logger.info("Analyzing data")
    df_analyzed = analyze_data(df, args)
    
    if df_analyzed is None or df_analyzed.empty:
        logger.error("Failed to analyze data")
        return
    
    logger.info(f"Successfully analyzed data")
    
    # Visualize data with terminal chart
    logger.info("Visualizing data with terminal chart")
    visualize(df_analyzed, args)
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
