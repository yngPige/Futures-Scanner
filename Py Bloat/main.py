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
from src.visualization.terminal_chart import TerminalChartGenerator
from src.utils.helpers import (
    create_directory, save_dataframe, load_dataframe,
    save_config, load_config, calculate_performance_metrics,
    calculate_trading_metrics
)

# Configure logging using custom logging utility
from src.utils.logging_utils import configure_logging

# Configure logging to only show errors in console and save to error log file
configure_logging()
logger = logging.getLogger(__name__)

# Create directories
create_directory('data')
create_directory('models')
create_directory('results')
create_directory('charts')
create_directory('error_logs')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3lacks Crypto Futures Scanner')

    # Mode arguments
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['fetch', 'analyze', 'train', 'predict', 'backtest', 'llm', 'all'],
                        help='Operation mode')

    # Data arguments
    parser.add_argument('--exchange', type=str, default='kraken',
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
    parser.add_argument('--use-llm', action='store_true', default=True,
                        help='Use LLM for analysis (default: enabled)')
    parser.add_argument('--llm-model', type=str, default='llama3-8b',
                        choices=['llama3-8b', 'llama3-70b', 'mistral-7b', 'phi3-mini',
                                 'fingpt-forecaster', 'hermes-llama3-financial', 'phi3-mini-financial', 'mistral-financial'],
                        help='LLM model to use')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU for LLM inference (default: enabled)')

    # Visualization arguments
    parser.add_argument('--theme', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='Chart theme')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive charts')
    parser.add_argument('--terminal-chart', action='store_true',
                        help='Display chart in terminal window')

    # Other arguments
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display charts')

    return parser.parse_args()

def fetch_data(args):
    """Fetch data from exchange."""
    logger.info(f"Fetching data for {args.symbol} from {args.exchange}")

    # Initialize data fetcher - try spot market first, then futures if needed
    fetcher = DataFetcher(exchange_id=args.exchange, timeframe=args.timeframe, market_type='spot')

    # Fetch data
    if '/' in args.symbol:
        # Fetch from exchange
        df = fetcher.fetch_ohlcv(args.symbol, limit=args.limit)

        # If spot market fetch failed, try futures market
        if df.empty:
            logger.info(f"Trying futures market for {args.symbol}")
            futures_fetcher = DataFetcher(exchange_id=args.exchange, timeframe=args.timeframe, market_type='future')
            df = futures_fetcher.fetch_ohlcv(args.symbol, limit=args.limit)
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

def visualize(df, args, metrics=None, llm_recommendation=None):
    """Generate visualizations."""
    logger.info("Generating visualizations")

    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty DataFrame provided, cannot create visualization")
        return None

    # Check if we should use terminal chart
    use_terminal_chart = args.terminal_chart if hasattr(args, 'terminal_chart') else False

    # Check if we have LLM recommendations and should use advanced chart
    use_advanced_chart = args.interactive and llm_recommendation and 'error' not in llm_recommendation

    if use_terminal_chart:
        # Initialize terminal chart generator
        terminal_chart_gen = TerminalChartGenerator(theme=args.theme)

        # Generate terminal charts
        if use_advanced_chart:
            # Use advanced chart with entry/exit suggestions
            logger.info("Creating advanced terminal chart with entry/exit suggestions")
            success = terminal_chart_gen.create_advanced_chart_with_suggestions(
                df,
                llm_analysis=llm_recommendation,
                title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
            )

            if not success:
                logger.warning("Failed to create advanced terminal chart")
                return None
        else:
            # Use regular candlestick chart
            logger.info("Creating terminal candlestick chart")
            success = terminal_chart_gen.create_candlestick_chart(
                df,
                title=f"{args.symbol} - {args.timeframe} Timeframe"
            )

            if not success:
                logger.warning("Failed to create terminal candlestick chart")
                return None
    else:
        # Initialize regular chart generator
        chart_gen = ChartGenerator(theme=args.theme)

        # Generate charts
        if args.interactive:
            if use_advanced_chart:
                # Use advanced chart with entry/exit suggestions
                logger.info("Creating advanced chart with entry/exit suggestions")
                fig = chart_gen.create_advanced_chart_with_suggestions(
                    df,
                    llm_analysis=llm_recommendation,
                    title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
                )
            else:
                # Use regular interactive chart
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
                chart_type = "advanced" if use_advanced_chart else "interactive"
                filename = f"charts/{args.symbol.replace('/', '_')}_{args.timeframe}_{chart_type}_{timestamp}.html"
                pio.write_html(fig, file=filename)
                logger.info(f"Saved {chart_type} chart to {filename}")

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

                    # Perform LLM analysis if enabled
                    recommendation = None
                    if args.use_llm:
                        logger.info("Performing LLM analysis...")
                        recommendation = llm_analysis(df_analyzed, args)

                        if recommendation and "error" not in recommendation:
                            logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
                            logger.info(f"Risk Assessment: {recommendation['risk']}")
                        else:
                            error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
                            logger.warning(f"LLM analysis failed: {error_msg}")
                            recommendation = None

                    # Visualize with LLM recommendations if available
                    visualize(df_predictions, args, llm_recommendation=recommendation)

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

                    # Perform LLM analysis if enabled
                    recommendation = None
                    if args.use_llm:
                        logger.info("Performing LLM analysis...")
                        recommendation = llm_analysis(df_analyzed, args)

                        if recommendation and "error" not in recommendation:
                            logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
                            logger.info(f"Risk Assessment: {recommendation['risk']}")
                        else:
                            error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
                            logger.warning(f"LLM analysis failed: {error_msg}")
                            recommendation = None

                    # Visualize with LLM recommendations if available
                    visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics), llm_recommendation=recommendation)

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

        if recommendation:
            # Check if it's a fallback recommendation
            if recommendation.get("model") == "FALLBACK_TECHNICAL_INDICATORS":
                logger.info("Using fallback technical indicator-based recommendation")
                logger.info(f"Fallback Recommendation: {recommendation['recommendation']}")
                logger.info(f"Risk Assessment: {recommendation['risk']}")
            elif "error" not in recommendation:
                logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
                logger.info(f"Risk Assessment: {recommendation['risk']}")
            else:
                error_msg = recommendation.get("error", "Unknown error")
                logger.error(f"LLM analysis failed: {error_msg}")
                recommendation = None

            # Generate visualization with entry/exit suggestions if we have a recommendation
            if recommendation:
                visualize(df_analyzed, args, llm_recommendation=recommendation)
                logger.info("Analysis complete. Check results file for details.")
            else:
                # Generate regular visualization without suggestions
                visualize(df_analyzed, args)
        else:
            logger.error("Failed to get any recommendation")
            # Generate regular visualization without suggestions
            visualize(df_analyzed, args)

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

        # Perform LLM analysis if enabled
        recommendation = None
        if args.use_llm:
            logger.info("Performing LLM analysis...")
            recommendation = llm_analysis(df_analyzed, args)

            if recommendation and "error" not in recommendation:
                logger.info(f"LLM Analysis Recommendation: {recommendation['recommendation']}")
                logger.info(f"Risk Assessment: {recommendation['risk']}")
            else:
                error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
                logger.warning(f"LLM analysis failed: {error_msg}")
                recommendation = None

        # Visualize with LLM recommendations if available
        visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics), llm_recommendation=recommendation)

        logger.info("Successfully completed all operations")

if __name__ == "__main__":
    main()
