"""
Test script for the LLM analysis and advanced chart with entry/exit suggestions.

This script downloads the LLM model if needed, fetches data, performs analysis,
runs LLM analysis, and displays an advanced chart with entry/exit suggestions.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
import plotly.io as pio

# Import the monkey patch first
import monkey_patch

from src.data.data_fetcher import DataFetcher
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS
from src.visualization.chart_generator import ChartGenerator
from src.utils.helpers import create_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test LLM Analysis and Advanced Chart')

    # Data arguments
    parser.add_argument('--exchange', type=str, default='coinbase',
                        help='Exchange to fetch data from')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for data')
    parser.add_argument('--limit', type=int, default=300,
                        help='Number of candles to fetch')

    # LLM arguments
    parser.add_argument('--llm-model', type=str, default='phi3-mini',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='LLM model to use')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for LLM inference')

    # Visualization arguments
    parser.add_argument('--theme', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='Chart theme')
    parser.add_argument('--save', action='store_true',
                        help='Save chart to file')

    return parser.parse_args()

def download_model(model_name):
    """Download the LLM model if it doesn't exist."""
    # Get model info
    model_info = AVAILABLE_MODELS.get(model_name, {})
    if not model_info:
        logger.error(f"Unknown model: {model_name}")
        return False

    # Get model path
    model_path = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
    create_directory(model_path)

    # Get model filename
    model_filename = model_info.get('name')
    if not model_filename:
        logger.error(f"No filename specified for model: {model_name}")
        return False

    # Check if model exists
    model_file_path = os.path.join(model_path, model_filename)
    if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
        logger.info(f"Model already exists at {model_file_path}")
        return True

    # Get model URL
    model_url = model_info.get('url')
    if not model_url:
        logger.error(f"No URL specified for model: {model_name}")
        return False

    # Download model
    logger.info(f"Downloading model {model_name} from {model_url}")
    logger.info(f"This may take a while depending on your internet connection...")

    try:
        import requests
        from tqdm import tqdm

        # Create a streaming request
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        # Create a progress bar
        progress_bar = tqdm(
            total=total_size, unit='B', unit_scale=True,
            desc=f"Downloading {model_filename}"
        )

        # Download the file
        with open(model_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()

        # Check if download was successful
        if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
            logger.info(f"Successfully downloaded model to {model_file_path}")
            return True
        else:
            logger.error(f"Failed to download model to {model_file_path}")
            return False

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    logger.info(f"Testing LLM analysis and advanced chart for {args.symbol} on {args.timeframe} timeframe")

    # Download model if needed
    if not download_model(args.llm_model):
        logger.error(f"Failed to download model {args.llm_model}")
        return

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

    # Set DataFrame attributes for LLM analysis
    df_analyzed.attrs["symbol"] = args.symbol
    df_analyzed.attrs["timeframe"] = args.timeframe

    # Perform LLM analysis
    logger.info(f"Performing LLM analysis with {args.llm_model} model")

    # Configure GPU usage
    n_gpu_layers = 0
    if args.use_gpu:
        logger.info("GPU acceleration enabled for LLM inference")
        n_gpu_layers = -1  # Use all layers on GPU

    # Initialize LLM analyzer
    llm_analyzer = LocalLLMAnalyzer(
        model_name=args.llm_model,
        n_gpu_layers=n_gpu_layers
    )

    # Analyze with LLM
    recommendation = llm_analyzer.analyze(df_analyzed)

    if recommendation and "error" not in recommendation:
        logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
        logger.info(f"Risk Assessment: {recommendation['risk']}")

        if 'entry_price' in recommendation:
            logger.info(f"Entry Price: {recommendation['entry_price']}")

        if 'stop_loss' in recommendation:
            logger.info(f"Stop Loss: {recommendation['stop_loss']}")

        if 'take_profit' in recommendation:
            logger.info(f"Take Profit: {recommendation['take_profit']}")
    else:
        error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
        logger.error(f"LLM analysis failed: {error_msg}")
        recommendation = None

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
        filename = f"charts/{args.symbol.replace('/', '_')}_{args.timeframe}_advanced_{timestamp}.html"
        pio.write_html(fig, file=filename)
        logger.info(f"Saved advanced chart to {filename}")

    # Display chart
    fig.show()

    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
