"""
3lacks Crypto Futures

This application parses cryptocurrency futures markets with a pretrained AI model
for optimized technical analysis to provide price movement predictions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.io as pio

# We're using a custom implementation of squeeze_pro to avoid numpy compatibility issues

from src.data.data_fetcher import DataFetcher
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.models.prediction_model import PredictionModel
from src.visualization.chart_generator import ChartGenerator
from src.utils.helpers import (
    create_directory, save_dataframe, load_dataframe,
    save_config, load_config, calculate_performance_metrics,
    calculate_trading_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
create_directory('data')
create_directory('models')
create_directory('results')
create_directory('charts')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3lacks Crypto Futures Scanner')

    # Mode arguments
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['fetch', 'analyze', 'train', 'predict', 'backtest', 'llm', 'all'],
                        help='Operation mode')

    # Data arguments
    parser.add_argument('--exchange', type=str, default='coinbase',
                        help='Exchange to fetch data from')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe for data')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Number of candles to fetch')

    # Model arguments
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting'],
                        help='Type of model to use')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model')
    parser.add_argument('--tune', action='store_true',
                        help='Tune model hyperparameters')

    # LLM arguments
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for analysis')
    parser.add_argument('--llm-model', type=str, default='llama3-8b',
                        choices=['llama3-8b', 'llama3-70b', 'mistral-7b', 'phi3-mini',
                                 'fingpt-forecaster', 'hermes-llama3-financial', 'phi3-mini-financial', 'mistral-financial'],
                        help='LLM model to use')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for LLM inference')

    # Visualization arguments
    parser.add_argument('--theme', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='Chart theme')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive charts')

    # Other arguments
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display charts')

    return parser.parse_args()

def fetch_data(args):
    """Fetch data from exchange."""
    logger.info(f"Fetching data for {args.symbol} from {args.exchange}")

    # Initialize data fetcher
    fetcher = DataFetcher(exchange_id=args.exchange, timeframe=args.timeframe)

    # Fetch data
    if '/' in args.symbol:
        # Fetch from exchange
        df = fetcher.fetch_ohlcv(args.symbol, limit=args.limit)
    else:
        # Fetch from yfinance as fallback
        df = fetcher.fetch_from_yfinance(args.symbol, period='1y', interval=args.timeframe)

    if df.empty:
        logger.error(f"Failed to fetch data for {args.symbol}")
        return None

    # Save data if requested
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/{args.symbol.replace('/', '_')}_{args.timeframe}_{timestamp}.csv"
        save_dataframe(df, filename)

    return df

def analyze_data(df, args):
    """Perform technical analysis on data."""
    logger.info("Performing technical analysis")

    # Initialize analyzer
    analyzer = TechnicalAnalyzer()

    # Add indicators
    df_with_indicators = analyzer.add_all_indicators(df)

    # Check if indicators were added successfully
    if df_with_indicators.empty:
        logger.error("Failed to add indicators, resulting DataFrame is empty")
        return None

    # Generate signals
    df_with_signals = analyzer.generate_signals(df_with_indicators)

    # Save results if requested
    if args.save and not df_with_signals.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{args.symbol.replace('/', '_')}_{args.timeframe}_analysis_{timestamp}.csv"
        save_dataframe(df_with_signals, filename)
        logger.info(f"Saved analysis results to {filename}")

    return df_with_signals

def train_model(df, args):
    """Train prediction model."""
    logger.info(f"Training {args.model_type} model")

    # Initialize model
    model = PredictionModel(model_dir='models')

    # Prepare features and target
    X, y = model.prepare_features_target(df)

    if X is None or y is None:
        logger.error("Failed to prepare features and target")
        return None

    # Train model
    success = model.train_model(X, y, model_type=args.model_type, tune_hyperparams=args.tune)

    if not success:
        logger.error("Failed to train model")
        return None

    # Save model if requested
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.symbol.replace('/', '_')}_{args.model_type}_{timestamp}"
        model_path = model.save_model(model_name)
        logger.info(f"Model saved to {model_path}")
        return model, model_path

    return model, None

def predict(df, model_path, args):
    """Make predictions using trained model."""
    logger.info("Making predictions")

    # Initialize model
    model = PredictionModel(model_dir='models')

    # Load model
    if model_path:
        success = model.load_model(model_path)
        if not success:
            logger.error(f"Failed to load model from {model_path}")
            return None
    else:
        logger.error("No model path provided")
        return None

    # Make predictions
    df_with_predictions = model.predict(df)

    if df_with_predictions is None:
        logger.error("Failed to make predictions")
        return None

    # Save results if requested
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{args.symbol.replace('/', '_')}_{args.timeframe}_predictions_{timestamp}.csv"
        save_dataframe(df_with_predictions, filename)

    return df_with_predictions

def backtest(df_with_predictions, args):
    """Backtest predictions."""
    logger.info("Backtesting predictions")

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(df_with_predictions)

    if performance_metrics:
        logger.info(f"Performance metrics: {performance_metrics}")

    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(df_with_predictions)

    if trading_metrics:
        logger.info(f"Trading metrics: {trading_metrics}")

    # Save results if requested
    if args.save and performance_metrics and trading_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{args.symbol.replace('/', '_')}_{args.timeframe}_backtest_{timestamp}.json"

        metrics = {
            'performance': performance_metrics,
            'trading': trading_metrics
        }

        save_config(metrics, filename)

    return performance_metrics, trading_metrics

def llm_analysis(df, args):
    """Perform LLM analysis on market data."""
    logger.info("Performing LLM analysis")

    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty DataFrame provided, cannot perform LLM analysis")
        return None

    try:
        # Import LLM analyzer
        from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS

        # Set DataFrame attributes for better context
        df.attrs["symbol"] = args.symbol
        df.attrs["timeframe"] = args.timeframe

        # Get model info
        model_info = AVAILABLE_MODELS.get(args.llm_model, {})
        model_name = model_info.get('name', args.llm_model)

        # Configure GPU usage
        n_gpu_layers = 0
        if args.use_gpu:
            logger.info("GPU acceleration enabled. Using GPU for inference.")
            n_gpu_layers = -1  # Use all layers on GPU

        # Initialize LLM analyzer
        logger.info(f"Initializing local LLM analyzer with model {args.llm_model}")
        llm_analyzer = LocalLLMAnalyzer(
            model_name=model_name,
            n_gpu_layers=n_gpu_layers
        )

        # Perform LLM analysis
        logger.info("Analyzing market data with local LLM...")
        recommendation = llm_analyzer.analyze(df)

        # Save analysis if requested
        if args.save and "error" not in recommendation:
            filename = llm_analyzer.save_analysis(recommendation, args.symbol, args.timeframe)
            if filename:
                logger.info(f"Saved LLM analysis to {filename}")

        return recommendation

    except Exception as e:
        logger.error(f"Error performing LLM analysis: {e}")
        return {"error": str(e)}

def visualize(df, args, metrics=None):
    """Generate visualizations."""
    logger.info("Generating visualizations")

    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty DataFrame provided, cannot create visualization")
        return None

    # Initialize chart generator
    chart_gen = ChartGenerator(theme=args.theme)

    # Generate charts
    if args.interactive:
        fig = chart_gen.create_interactive_chart(
            df,
            title=f"{args.symbol} - {args.timeframe} Timeframe"
        )

        # Check if figure was created successfully
        if fig is None:
            logger.warning("Failed to create interactive chart")
            return None

        # Save chart if requested
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"charts/{args.symbol.replace('/', '_')}_{args.timeframe}_interactive_{timestamp}.html"
            pio.write_html(fig, file=filename)
            logger.info(f"Saved interactive chart to {filename}")

        # Display chart if requested
        if not args.no_display:
            fig.show()
    else:
        fig = chart_gen.plot_price_with_indicators(
            df,
            title=f"{args.symbol} - {args.timeframe} Timeframe"
        )

        # Check if figure was created successfully
        if fig is None:
            logger.warning("Failed to create chart")
            return None

        # Save chart if requested
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"charts/{args.symbol.replace('/', '_')}_{args.timeframe}_{timestamp}.png"
            fig.savefig(filename)
            logger.info(f"Saved chart to {filename}")

        # Display chart if requested
        if not args.no_display:
            plt.show()

    return fig

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Execute based on mode
    if args.mode == 'fetch':
        df = fetch_data(args)
        if df is not None:
            logger.info(f"Successfully fetched {len(df)} rows of data")

    elif args.mode == 'analyze':
        df = fetch_data(args)
        if df is not None:
            df_analyzed = analyze_data(df, args)
            if df_analyzed is not None and not df_analyzed.empty:
                logger.info(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators")
                fig = visualize(df_analyzed, args)
                if fig is None:
                    logger.warning("Visualization failed, but analysis was successful")
            else:
                logger.warning("Analysis resulted in empty DataFrame or failed")

    elif args.mode == 'train':
        df = fetch_data(args)
        if df is not None:
            df_analyzed = analyze_data(df, args)
            if df_analyzed is not None:
                model, model_path = train_model(df_analyzed, args)
                if model is not None:
                    logger.info("Successfully trained model")

    elif args.mode == 'predict':
        if args.model_path is None:
            logger.error("Model path is required for prediction mode")
            return

        df = fetch_data(args)
        if df is not None:
            df_analyzed = analyze_data(df, args)
            if df_analyzed is not None:
                df_predictions = predict(df_analyzed, args.model_path, args)
                if df_predictions is not None:
                    logger.info("Successfully made predictions")
                    visualize(df_predictions, args)

    elif args.mode == 'backtest':
        if args.model_path is None:
            logger.error("Model path is required for backtest mode")
            return

        df = fetch_data(args)
        if df is not None:
            df_analyzed = analyze_data(df, args)
            if df_analyzed is not None:
                df_predictions = predict(df_analyzed, args.model_path, args)
                if df_predictions is not None:
                    performance_metrics, trading_metrics = backtest(df_predictions, args)
                    visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics))

    elif args.mode == 'llm':
        # Check if LLM analysis is enabled
        if not args.use_llm:
            logger.error("LLM analysis is not enabled. Use --use-llm flag.")
            return

        # Fetch data
        df = fetch_data(args)
        if df is None:
            return

        # Analyze data
        df_analyzed = analyze_data(df, args)
        if df_analyzed is None:
            return

        # Perform LLM analysis
        recommendation = llm_analysis(df_analyzed, args)

        if recommendation and "error" not in recommendation:
            logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
            logger.info(f"Risk Assessment: {recommendation['risk']}")
            logger.info("Analysis complete. Check results file for details.")
        else:
            error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
            logger.error(f"LLM analysis failed: {error_msg}")

    elif args.mode == 'all':
        # Fetch data
        df = fetch_data(args)
        if df is None:
            return

        # Analyze data
        df_analyzed = analyze_data(df, args)
        if df_analyzed is None:
            return

        # Train model
        model, model_path = train_model(df_analyzed, args)
        if model is None:
            return

        # Make predictions
        df_predictions = predict(df_analyzed, model_path, args)
        if df_predictions is None:
            return

        # Backtest
        performance_metrics, trading_metrics = backtest(df_predictions, args)

        # Visualize
        visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics))

        # Perform LLM analysis if enabled
        if args.use_llm:
            logger.info("Performing LLM analysis...")
            recommendation = llm_analysis(df_analyzed, args)

            if recommendation and "error" not in recommendation:
                logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
                logger.info(f"Risk Assessment: {recommendation['risk']}")
            else:
                error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
                logger.warning(f"LLM analysis failed: {error_msg}")

        logger.info("Successfully completed all operations")

if __name__ == "__main__":
    main()
