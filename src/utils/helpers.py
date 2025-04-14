"""
Helper Utilities for Crypto Futures Scanner

This module provides utility functions for the Crypto Futures Scanner application.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
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
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save DataFrame
        if format.lower() == 'csv':
            df.to_csv(filepath)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

        logger.info(f"Saved DataFrame to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {e}")
        return False

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
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            else:
                logger.error(f"Could not infer format from extension: {ext}")
                return None

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
