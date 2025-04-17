"""
Helper Utilities for Crypto Futures Scanner

This module provides utility functions for the Crypto Futures Scanner application.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
import hashlib
from datetime import datetime, timedelta

# Configure logging - only show errors
logging.basicConfig(
    level=logging.ERROR,  # Only log errors and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): Directory path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def save_dataframe(df, filepath, format='csv'):
    """
    Save a DataFrame to a file.

    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the file
        format (str): File format ('csv', 'parquet', or 'pickle')

    Returns:
        str: Path to the saved file if successful, None otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Check if filepath has the correct extension
        ext = os.path.splitext(filepath)[1].lower()
        if format.lower() == 'csv' and ext != '.csv':
            filepath = f"{filepath}.csv"
        elif format.lower() == 'parquet' and ext != '.parquet':
            filepath = f"{filepath}.parquet"
        elif format.lower() == 'pickle' and ext not in ['.pkl', '.pickle']:
            filepath = f"{filepath}.pkl"

        # Save DataFrame
        if format.lower() == 'csv':
            df.to_csv(filepath)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return None

        logger.info(f"Saved DataFrame to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {e}")
        return None

def load_dataframe(filepath, format=None):
    """
    Load a DataFrame from a file.

    Args:
        filepath (str): Path to the file
        format (str, optional): File format ('csv', 'parquet', or 'pickle')
            If None, infer from file extension

    Returns:
        pd.DataFrame: Loaded DataFrame, or None if error
    """
    try:
        # Infer format from file extension if not provided
        if format is None:
            # Check if filepath has an extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            else:
                # If no extension or unrecognized extension, assume it's a CSV file
                # and append .csv to the filepath
                logger.warning(f"No recognized extension in {filepath}, assuming CSV format")
                filepath = f"{filepath}.csv"
                format = 'csv'

        # Load DataFrame
        if format.lower() == 'csv':
            df = pd.read_csv(filepath)
            # Convert timestamp column to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        elif format.lower() == 'parquet':
            df = pd.read_parquet(filepath)
        elif format.lower() == 'pickle':
            df = pd.read_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return None

        logger.info(f"Loaded DataFrame from {filepath} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filepath}: {e}")
        return None

def save_config(config, filepath):
    """
    Save configuration to a JSON file.

    Args:
        config (dict): Configuration dictionary
        filepath (str): Path to save the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save config
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info(f"Saved configuration to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {filepath}: {e}")
        return False

def load_config(filepath):
    """
    Load configuration from a JSON file.

    Args:
        filepath (str): Path to the file

    Returns:
        dict: Configuration dictionary, or None if error
    """
    try:
        # Load config
        with open(filepath, 'r') as f:
            config = json.load(f)

        logger.info(f"Loaded configuration from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {filepath}: {e}")
        return None

def calculate_performance_metrics(df, prediction_col='prediction', actual_col='target'):
    """
    Calculate performance metrics for predictions.

    Args:
        df (pd.DataFrame): DataFrame with predictions and actual values
        prediction_col (str): Name of the prediction column
        actual_col (str): Name of the actual value column

    Returns:
        dict: Dictionary with performance metrics
    """
    try:
        if prediction_col not in df.columns or actual_col not in df.columns:
            logger.error(f"Missing columns: {prediction_col} or {actual_col}")
            return None

        # Calculate metrics
        true_positives = ((df[prediction_col] == 1) & (df[actual_col] == 1)).sum()
        false_positives = ((df[prediction_col] == 1) & (df[actual_col] == 0)).sum()
        true_negatives = ((df[prediction_col] == 0) & (df[actual_col] == 0)).sum()
        false_negatives = ((df[prediction_col] == 0) & (df[actual_col] == 1)).sum()

        # Calculate derived metrics
        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

        logger.info(f"Calculated performance metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return None

def calculate_trading_metrics(df, prediction_col='prediction', price_col='close'):
    """
    Calculate trading performance metrics based on predictions.

    Args:
        df (pd.DataFrame): DataFrame with predictions and price data
        prediction_col (str): Name of the prediction column
        price_col (str): Name of the price column

    Returns:
        dict: Dictionary with trading metrics
    """
    try:
        if prediction_col not in df.columns or price_col not in df.columns:
            logger.error(f"Missing columns: {prediction_col} or {price_col}")
            return None

        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Calculate returns
        df_copy['next_return'] = df_copy[price_col].pct_change(1).shift(-1)

        # Calculate strategy returns (long when prediction is 1, cash when prediction is 0)
        df_copy['strategy_return'] = df_copy['next_return'] * df_copy[prediction_col]

        # Calculate buy and hold returns
        df_copy['buy_hold_return'] = df_copy['next_return']

        # Calculate cumulative returns
        df_copy['cum_strategy_return'] = (1 + df_copy['strategy_return']).cumprod() - 1
        df_copy['cum_buy_hold_return'] = (1 + df_copy['buy_hold_return']).cumprod() - 1

        # Calculate metrics
        total_trades = df_copy[prediction_col].diff().abs().sum()
        winning_trades = ((df_copy['strategy_return'] > 0) & (df_copy[prediction_col] == 1)).sum()
        losing_trades = ((df_copy['strategy_return'] < 0) & (df_copy[prediction_col] == 1)).sum()

        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0

        strategy_return = df_copy['cum_strategy_return'].iloc[-1] if len(df_copy) > 0 else 0
        buy_hold_return = df_copy['cum_buy_hold_return'].iloc[-1] if len(df_copy) > 0 else 0

        # Calculate annualized returns
        days = (df_copy.index[-1] - df_copy.index[0]).days if isinstance(df_copy.index, pd.DatetimeIndex) and len(df_copy) > 1 else 365
        years = days / 365

        annualized_strategy_return = (1 + strategy_return) ** (1 / years) - 1 if years > 0 else 0
        annualized_buy_hold_return = (1 + buy_hold_return) ** (1 / years) - 1 if years > 0 else 0

        # Calculate max drawdown
        strategy_cumulative = (1 + df_copy['strategy_return']).cumprod()
        buy_hold_cumulative = (1 + df_copy['buy_hold_return']).cumprod()

        strategy_max_drawdown = (strategy_cumulative / strategy_cumulative.cummax() - 1).min() if len(strategy_cumulative) > 0 else 0
        buy_hold_max_drawdown = (buy_hold_cumulative / buy_hold_cumulative.cummax() - 1).min() if len(buy_hold_cumulative) > 0 else 0

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        strategy_sharpe = df_copy['strategy_return'].mean() / df_copy['strategy_return'].std() * np.sqrt(252) if df_copy['strategy_return'].std() > 0 else 0
        buy_hold_sharpe = df_copy['buy_hold_return'].mean() / df_copy['buy_hold_return'].std() * np.sqrt(252) if df_copy['buy_hold_return'].std() > 0 else 0

        # Create metrics dictionary
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'annualized_strategy_return': annualized_strategy_return,
            'annualized_buy_hold_return': annualized_buy_hold_return,
            'strategy_max_drawdown': strategy_max_drawdown,
            'buy_hold_max_drawdown': buy_hold_max_drawdown,
            'strategy_sharpe': strategy_sharpe,
            'buy_hold_sharpe': buy_hold_sharpe
        }

        logger.info(f"Calculated trading metrics: Win Rate={win_rate:.4f}, Strategy Return={strategy_return:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating trading metrics: {e}")
        return None


def get_cache_directory():
    """
    Get the cache directory for the application.

    Returns:
        str: Path to the cache directory
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner")
    create_directory(cache_dir)
    return cache_dir

def get_analysis_cache_path(symbol, exchange, timeframe):
    """
    Get the path to the cached analysis file for a specific symbol, exchange, and timeframe.

    Args:
        symbol (str): Trading symbol
        exchange (str): Exchange name
        timeframe (str): Timeframe of the data

    Returns:
        str: Path to the cached analysis file
    """
    cache_dir = get_cache_directory()
    analysis_cache_dir = os.path.join(cache_dir, "analysis_cache")
    create_directory(analysis_cache_dir)

    # Create a filename based on the symbol, exchange, and timeframe
    filename = f"{symbol.replace('/', '_')}_{exchange}_{timeframe}_analysis.pkl"
    return os.path.join(analysis_cache_dir, filename)

def save_analysis_to_cache(df, symbol, exchange, timeframe, max_age_hours=24):
    """
    Save analysis results to cache.

    Args:
        df (pd.DataFrame): DataFrame with analysis results
        symbol (str): Trading symbol
        exchange (str): Exchange name
        timeframe (str): Timeframe of the data
        max_age_hours (int): Maximum age of the cache in hours

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a dictionary with the analysis data and metadata
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'max_age_hours': max_age_hours,
            'data': df
        }

        # Get the cache file path
        cache_path = get_analysis_cache_path(symbol, exchange, timeframe)

        # Save to pickle file
        with open(cache_path, 'wb') as f:
            pd.to_pickle(cache_data, f)

        logger.info(f"Saved analysis to cache: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis to cache: {e}")
        return False

def load_analysis_from_cache(symbol, exchange, timeframe):
    """
    Load analysis results from cache if available and not expired.

    Args:
        symbol (str): Trading symbol
        exchange (str): Exchange name
        timeframe (str): Timeframe of the data

    Returns:
        pd.DataFrame: DataFrame with analysis results, or None if not available or expired
    """
    try:
        # Get the cache file path
        cache_path = get_analysis_cache_path(symbol, exchange, timeframe)

        # Check if cache file exists
        if not os.path.exists(cache_path):
            logger.info(f"No cache file found for {symbol} on {exchange} ({timeframe})")
            return None

        # Load cache data
        with open(cache_path, 'rb') as f:
            cache_data = pd.read_pickle(f)

        # Check if cache is expired
        cache_timestamp = datetime.fromisoformat(cache_data['timestamp'])
        max_age_hours = cache_data.get('max_age_hours', 24)
        cache_age = datetime.now() - cache_timestamp

        if cache_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {symbol} on {exchange} ({timeframe})")
            return None

        logger.info(f"Loaded analysis from cache: {cache_path}")
        return cache_data['data']
    except Exception as e:
        logger.error(f"Error loading analysis from cache: {e}")
        return None

def get_settings_path():
    """
    Get the path to the settings file.

    Returns:
        str: Path to the settings file
    """
    cache_dir = get_cache_directory()
    return os.path.join(cache_dir, "settings.json")

def save_settings(settings):
    """
    Save application settings to a file.

    Args:
        settings (dict): Settings dictionary

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings_path = get_settings_path()
        return save_config(settings, settings_path)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

def load_settings():
    """
    Load application settings from a file.

    Returns:
        dict: Settings dictionary, or None if error
    """
    try:
        settings_path = get_settings_path()
        if not os.path.exists(settings_path):
            logger.info(f"No settings file found at {settings_path}")
            return None
        return load_config(settings_path)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return None

def get_previous_analyses():
    """
    Get a list of previous analyses from the results directory.

    Returns:
        dict: Dictionary with symbol/timeframe as keys and lists of analysis files as values
    """
    try:
        # Create results directory if it doesn't exist
        create_directory('results')

        # Dictionary to store analyses by symbol and timeframe
        analyses = {}

        # List all files in the results directory
        for filename in os.listdir('results'):
            # Skip directories
            if os.path.isdir(os.path.join('results', filename)):
                continue

            # Parse filename to extract symbol, timeframe, and type
            parts = filename.split('_')

            # Skip files that don't match our naming pattern
            if len(parts) < 3:
                continue

            # Handle symbol with underscore (e.g., BTC_USDT from BTC/USDT)
            if len(parts) >= 4 and parts[1] in ['USDT', 'USD', 'BUSD', 'USDC']:
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = parts[2]
                file_type = '_'.join(parts[3:]).split('.')[0]
                timestamp = None

                # Extract timestamp if present
                for part in parts[3:]:
                    if len(part) == 8 and part.isdigit():
                        timestamp = part
                    elif len(part) == 14 and part.isdigit():
                        timestamp = part
            else:
                symbol = parts[0]
                timeframe = parts[1]
                file_type = '_'.join(parts[2:]).split('.')[0]
                timestamp = None

                # Extract timestamp if present
                for part in parts[2:]:
                    if len(part) == 8 and part.isdigit():
                        timestamp = part
                    elif len(part) == 14 and part.isdigit():
                        timestamp = part

            # Create a key for the symbol and timeframe
            key = f"{symbol}_{timeframe}"

            # Add to the dictionary
            if key not in analyses:
                analyses[key] = []

            # Add file info
            file_info = {
                'filename': filename,
                'path': os.path.join('results', filename),
                'symbol': symbol,
                'timeframe': timeframe,
                'type': file_type,
                'timestamp': timestamp,
                'datetime': None
            }

            # Try to parse the timestamp
            if timestamp:
                try:
                    if len(timestamp) == 8:
                        # Format: YYYYMMDD
                        file_info['datetime'] = datetime.strptime(timestamp, '%Y%m%d')
                    elif len(timestamp) == 14:
                        # Format: YYYYMMDD_HHMMSS
                        file_info['datetime'] = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                except ValueError:
                    pass

            # Get file modification time as fallback
            if file_info['datetime'] is None:
                file_info['datetime'] = datetime.fromtimestamp(
                    os.path.getmtime(os.path.join('results', filename))
                )

            analyses[key].append(file_info)

        # Sort each list by datetime (most recent first)
        for key in analyses:
            analyses[key].sort(key=lambda x: x['datetime'], reverse=True)

        return analyses
    except Exception as e:
        logger.error(f"Error getting previous analyses: {e}")
        return {}

def load_previous_analysis(file_path):
    """
    Load a previous analysis from a file.

    Args:
        file_path (str): Path to the analysis file

    Returns:
        tuple: (DataFrame or dict, file_type) - The loaded analysis data and the type of file
    """
    try:
        # Get the file extension
        ext = os.path.splitext(file_path)[1].lower()

        # Load based on file type
        if ext == '.csv':
            # Load DataFrame
            df = load_dataframe(file_path)
            return df, 'dataframe'
        elif ext == '.json':
            # Load JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data, 'json'
        elif ext == '.txt':
            # Load text
            with open(file_path, 'r') as f:
                data = f.read()
            return data, 'text'
        elif ext in ['.pkl', '.pickle']:
            # Load pickle
            with open(file_path, 'rb') as f:
                data = pd.read_pickle(f)
            return data, 'pickle'
        else:
            logger.error(f"Unsupported file type: {ext}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading previous analysis: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'close': np.random.normal(100, 5, 100),
        'prediction': np.random.choice([0, 1], 100),
        'target': np.random.choice([0, 1], 100)
    }, index=dates)

    # Save and load DataFrame
    save_dataframe(df, 'sample_data.csv')
    loaded_df = load_dataframe('sample_data.csv')

    # Calculate performance metrics
    metrics = calculate_performance_metrics(df)
    print(metrics)

    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(df)
    print(trading_metrics)

    # Test caching functions
    save_analysis_to_cache(df, 'BTC/USDT', 'kraken', '1h')
    cached_df = load_analysis_from_cache('BTC/USDT', 'kraken', '1h')
    print(f"Cached DataFrame loaded: {cached_df is not None}")
