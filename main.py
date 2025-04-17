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
                        choices=['fetch', 'analyze', 'train', 'predict', 'backtest', 'all'],
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
                        help='Tune hyperparameters')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')

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
        filepath = os.path.join('data', f"{args.symbol.replace('/', '_')}_{args.timeframe}")
        saved_path = save_dataframe(df, filepath)
        if saved_path:
            logger.info(f"Saved data to {saved_path}")

    return df

def analyze_data(df, args):
    """Analyze data using technical indicators."""
    logger.info("Analyzing data with technical indicators")

    # Initialize analyzer
    analyzer = TechnicalAnalyzer()

    # Analyze data
    df_analyzed = analyzer.analyze(df)

    if df_analyzed is None or df_analyzed.empty:
        logger.error("Failed to analyze data")
        return None

    # Save analyzed data if requested
    if args.save:
        filepath = os.path.join('results', f"{args.symbol.replace('/', '_')}_{args.timeframe}_analyzed")
        saved_path = save_dataframe(df_analyzed, filepath)
        if saved_path:
            logger.info(f"Saved analyzed data to {saved_path}")

    return df_analyzed

def train_model(df, args):
    """Train prediction model."""
    logger.info(f"Training {args.model_type} model")

    # Initialize model
    model = PredictionModel(model_dir='models')

    # Prepare features and target
    X, y = model.prepare_features_target(df)

    if X is None or y is None:
        logger.error("Failed to prepare features and target")
        return None, None

    # Train model
    success = model.train_model(X, y, model_type=args.model_type, tune_hyperparams=args.tune)

    if not success:
        logger.error("Failed to train model")
        return None, None

    # Save model
    model_path = model.save_model(f"{args.symbol.replace('/', '_')}_{args.timeframe}_{args.model_type}")
    if model_path is None:
        logger.error("Failed to save model")
        return model, None

    logger.info(f"Saved model to {model_path}")
    return model, model_path

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

    # Save predictions if requested
    if args.save:
        filepath = os.path.join('results', f"{args.symbol.replace('/', '_')}_{args.timeframe}_predictions")
        saved_path = save_dataframe(df_with_predictions, filepath)
        if saved_path:
            logger.info(f"Saved predictions to {saved_path}")

    return df_with_predictions

def backtest(df, args):
    """Backtest trading strategy."""
    logger.info("Backtesting trading strategy")

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(df)
    if performance_metrics is None:
        logger.error("Failed to calculate performance metrics")
        return None, None

    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(df)
    if trading_metrics is None:
        logger.error("Failed to calculate trading metrics")
        return performance_metrics, None

    # Save metrics if requested
    if args.save:
        # Save performance metrics
        perf_filename = os.path.join('results', f"{args.symbol.replace('/', '_')}_{args.timeframe}_performance.txt")
        with open(perf_filename, 'w') as f:
            for key, value in performance_metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved performance metrics to {perf_filename}")

        # Save trading metrics
        trade_filename = os.path.join('results', f"{args.symbol.replace('/', '_')}_{args.timeframe}_trading.txt")
        with open(trade_filename, 'w') as f:
            for key, value in trading_metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved trading metrics to {trade_filename}")

    return performance_metrics, trading_metrics

def llm_analysis(df, args):
    """Perform LLM analysis on market data."""
    logger.info("Performing LLM analysis on market data")

    try:
        # Import LLM analyzer
        from src.analysis.llm_analysis import LLMAnalyzer

        # Initialize LLM analyzer
        llm_analyzer = LLMAnalyzer()

        # Perform analysis
        analysis_results = llm_analyzer.analyze_market(df, args.symbol, args.timeframe)

        # If there was an error with Ollama, try the fallback analysis
        if 'error' in analysis_results and 'Ollama not available' in analysis_results['error']:
            logger.warning("Ollama not available, using fallback analysis")
            analysis_results = llm_analyzer.get_fallback_analysis(df, args.symbol, args.timeframe)

        # Save analysis if requested
        if args.save and 'analysis' in analysis_results:
            # Create results directory if it doesn't exist
            create_directory('results')

            # Save analysis to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join('results', f"{args.symbol.replace('/', '_')}_{args.timeframe}_llm_analysis_{timestamp}.txt")

            with open(filename, 'w') as f:
                f.write(f"LLM Analysis for {args.symbol} ({args.timeframe})\n")
                f.write(f"Timestamp: {analysis_results.get('timestamp', 'N/A')}\n")
                f.write(f"Sentiment: {analysis_results.get('sentiment', 'N/A')}\n")
                f.write(f"Risk: {analysis_results.get('risk', 'N/A')}\n\n")
                f.write(analysis_results['analysis'])

            logger.info(f"Saved LLM analysis to {filename}")

        return analysis_results

    except Exception as e:
        logger.error(f"Error performing LLM analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def visualize(df, args, metrics=None, llm_recommendation=None):
    """Generate visualizations."""
    logger.info("Generating visualizations")

    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty DataFrame provided, cannot create visualization")
        return None

    # Determine if we should use advanced chart with entry/exit suggestions
    use_advanced_chart = False
    if 'prediction' in df.columns or llm_recommendation is not None:
        use_advanced_chart = True

    # Generate browser chart
    chart_gen = ChartGenerator()

    # Generate terminal chart
    terminal_chart_gen = TerminalChartGenerator()

    # Determine chart type based on display mode
    if args.no_display:
        # Save chart without displaying
        if use_advanced_chart:
            # Use advanced chart with entry/exit suggestions
            logger.info("Creating advanced chart with entry/exit suggestions")
            fig = chart_gen.create_advanced_chart_with_suggestions(
                df,
                llm_analysis=llm_recommendation,
                title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
            )
        else:
            # Use regular candlestick chart
            logger.info("Creating regular candlestick chart")
            fig = chart_gen.create_candlestick_chart(
                df,
                title=f"{args.symbol} - {args.timeframe} Timeframe"
            )

        # Save chart if requested
        if args.save and fig is not None:
            filename = os.path.join('charts', f"{args.symbol.replace('/', '_')}_{args.timeframe}_chart.html")
            pio.write_html(fig, file=filename, auto_open=False)
            logger.info(f"Saved chart to {filename}")

        return fig
    else:
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

        # Save chart if requested
        if args.save:
            if use_advanced_chart:
                # Use advanced chart with entry/exit suggestions
                fig = chart_gen.create_advanced_chart_with_suggestions(
                    df,
                    llm_analysis=llm_recommendation,
                    title=f"{args.symbol} - {args.timeframe} Timeframe with Trading Levels"
                )
            else:
                # Use regular candlestick chart
                fig = chart_gen.create_candlestick_chart(
                    df,
                    title=f"{args.symbol} - {args.timeframe} Timeframe"
                )

            if fig is not None:
                filename = os.path.join('charts', f"{args.symbol.replace('/', '_')}_{args.timeframe}_chart.html")
                pio.write_html(fig, file=filename, auto_open=False)
                logger.info(f"Saved chart to {filename}")

        return None

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

                # Perform LLM analysis
                llm_results = llm_analysis(df_analyzed, args)

                # Visualize with LLM results
                fig = visualize(df_analyzed, args, llm_recommendation=llm_results)
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

                    # Perform LLM analysis
                    llm_results = llm_analysis(df_analyzed, args)

                    # Visualize with LLM results
                    visualize(df_predictions, args, llm_recommendation=llm_results)

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

                    # Perform LLM analysis
                    llm_results = llm_analysis(df_analyzed, args)

                    # Visualize with LLM results
                    visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics), llm_recommendation=llm_results)

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

        # Perform LLM analysis
        llm_results = llm_analysis(df_analyzed, args)

        # Visualize with LLM results
        visualize(df_predictions, args, metrics=(performance_metrics, trading_metrics), llm_recommendation=llm_results)

if __name__ == "__main__":
    main()
