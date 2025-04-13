"""
Test script for the advanced chart with mock entry/exit suggestions.

This script fetches data, performs analysis, generates mock LLM recommendations,
and displays an advanced chart with entry/exit suggestions.
"""

import argparse
import logging
import random
from datetime import datetime
import plotly.io as pio

# Import the monkey patch first
import monkey_patch

from src.data.data_fetcher import DataFetcher
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.visualization.chart_generator import ChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Advanced Chart with Mock Entry/Exit Suggestions')
    
    # Data arguments
    parser.add_argument('--exchange', type=str, default='coinbase',
                        help='Exchange to fetch data from')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for data')
    parser.add_argument('--limit', type=int, default=300,
                        help='Number of candles to fetch')
    
    # Visualization arguments
    parser.add_argument('--theme', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='Chart theme')
    parser.add_argument('--save', action='store_true',
                        help='Save chart to file')
    
    return parser.parse_args()

def generate_mock_recommendation(df):
    """Generate mock LLM recommendations for testing."""
    # Get the latest price
    latest_price = df['close'].iloc[-1]
    
    # Generate random recommendation
    recommendations = ['STRONG BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG SELL']
    recommendation = random.choice(recommendations)
    
    # Generate random risk level
    risks = ['LOW', 'MEDIUM', 'HIGH']
    risk = random.choice(risks)
    
    # Generate entry price (close to current price)
    entry_price = latest_price * random.uniform(0.98, 1.02)
    
    # Generate stop loss and take profit based on recommendation
    if recommendation in ['STRONG BUY', 'BUY']:
        # For buy recommendations, stop loss is below entry, take profit is above
        stop_loss = entry_price * random.uniform(0.95, 0.98)
        take_profit = entry_price * random.uniform(1.05, 1.10)
    elif recommendation in ['STRONG SELL', 'SELL']:
        # For sell recommendations, stop loss is above entry, take profit is below
        stop_loss = entry_price * random.uniform(1.02, 1.05)
        take_profit = entry_price * random.uniform(0.90, 0.95)
    else:
        # For neutral, set both randomly
        stop_loss = entry_price * random.uniform(0.96, 0.98)
        take_profit = entry_price * random.uniform(1.02, 1.04)
    
    # Calculate risk-reward ratio
    risk_amount = abs(entry_price - stop_loss)
    reward_amount = abs(take_profit - entry_price)
    risk_reward = reward_amount / risk_amount if risk_amount > 0 else 1.0
    
    # Create mock recommendation
    mock_recommendation = {
        'recommendation': recommendation,
        'risk': risk,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'analysis': f"Mock analysis for {df.attrs.get('symbol', 'Unknown')} on {df.attrs.get('timeframe', 'Unknown')} timeframe.",
        'timestamp': datetime.now().isoformat(),
        'model': 'Mock LLM Model'
    }
    
    return mock_recommendation

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    logger.info(f"Testing advanced chart with mock recommendations for {args.symbol} on {args.timeframe} timeframe")
    
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
    
    # Set DataFrame attributes for mock analysis
    df_analyzed.attrs["symbol"] = args.symbol
    df_analyzed.attrs["timeframe"] = args.timeframe
    
    # Generate mock recommendations
    logger.info("Generating mock LLM recommendations")
    recommendation = generate_mock_recommendation(df_analyzed)
    
    logger.info(f"Mock Recommendation: {recommendation['recommendation']}")
    logger.info(f"Risk Assessment: {recommendation['risk']}")
    logger.info(f"Entry Price: {recommendation['entry_price']:.2f}")
    logger.info(f"Stop Loss: {recommendation['stop_loss']:.2f}")
    logger.info(f"Take Profit: {recommendation['take_profit']:.2f}")
    logger.info(f"Risk-Reward Ratio: {recommendation['risk_reward']:.2f}")
    
    # Create advanced chart
    logger.info("Creating advanced chart with entry/exit suggestions")
    chart_gen = ChartGenerator(theme=args.theme)
    fig = chart_gen.create_advanced_chart_with_suggestions(
        df_analyzed,
        llm_analysis=recommendation,
        title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
    )
    
    if fig is None:
        logger.error("Failed to create advanced chart")
        return
    
    # Save chart if requested
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"charts/{args.symbol.replace('/', '_')}_{args.timeframe}_mock_{timestamp}.html"
        pio.write_html(fig, file=filename)
        logger.info(f"Saved mock chart to {filename}")
    
    # Display chart
    fig.show()
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
