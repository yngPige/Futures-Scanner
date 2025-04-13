"""
Test script for terminal-based charts.

This script fetches data, performs analysis, and displays charts in a terminal window.
"""

import argparse
import logging
from datetime import datetime

# Import the monkey patch first
import monkey_patch

from src.data.data_fetcher import DataFetcher
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.visualization.terminal_chart import TerminalChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Terminal Charts')
    
    # Data arguments
    parser.add_argument('--exchange', type=str, default='coinbase',
                        help='Exchange to fetch data from')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for data')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of candles to fetch')
    
    # Chart type arguments
    parser.add_argument('--chart-type', type=str, default='candlestick',
                        choices=['price', 'candlestick', 'advanced'],
                        help='Type of chart to display')
    
    # Visualization arguments
    parser.add_argument('--theme', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='Chart theme')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    logger.info(f"Testing terminal chart for {args.symbol} on {args.timeframe} timeframe")
    
    # Fetch data
    logger.info(f"Fetching data for {args.symbol} from {args.exchange}")
    fetcher = DataFetcher(exchange_id=args.exchange, timeframe=args.timeframe)
    df = fetcher.fetch_ohlcv(args.symbol, limit=args.limit)
    
    if df.empty:
        logger.error(f"Failed to fetch data for {args.symbol}")
        return
    
    logger.info(f"Successfully fetched {len(df)} candles")
    
    # Perform technical analysis
    logger.info("Performing technical analysis")
    analyzer = TechnicalAnalyzer()
    df_analyzed = analyzer.add_all_indicators(df)
    df_analyzed = analyzer.generate_signals(df_analyzed)
    
    if df_analyzed.empty:
        logger.error("Failed to perform technical analysis")
        return
    
    logger.info(f"Successfully added {len(df_analyzed.columns) - len(df.columns)} indicators")
    
    # Initialize terminal chart generator
    logger.info(f"Initializing terminal chart generator with {args.theme} theme")
    chart_gen = TerminalChartGenerator(theme=args.theme)
    
    # Display chart based on chart type
    if args.chart_type == 'price':
        logger.info("Displaying price chart in terminal")
        chart_gen.create_price_chart(
            df_analyzed,
            title=f"{args.symbol} - {args.timeframe} Timeframe"
        )
    elif args.chart_type == 'candlestick':
        logger.info("Displaying candlestick chart in terminal")
        chart_gen.create_candlestick_chart(
            df_analyzed,
            title=f"{args.symbol} - {args.timeframe} Timeframe"
        )
    elif args.chart_type == 'advanced':
        logger.info("Displaying advanced chart in terminal")
        
        # Create mock LLM analysis
        mock_analysis = {
            'recommendation': 'BUY',
            'risk': 'MEDIUM',
            'entry_price': df_analyzed['close'].iloc[-1] * 0.99,  # 1% below current price
            'stop_loss': df_analyzed['close'].iloc[-1] * 0.95,     # 5% below current price
            'take_profit': df_analyzed['close'].iloc[-1] * 1.05    # 5% above current price
        }
        
        chart_gen.create_advanced_chart_with_suggestions(
            df_analyzed,
            llm_analysis=mock_analysis,
            title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
        )
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
