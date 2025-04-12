"""
Technical Analysis Module for Crypto Futures Scanner

This module provides functionality to perform technical analysis on cryptocurrency data
using pandas-ta and other custom indicators.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

# Import custom indicators
from src.analysis.custom_indicators import squeeze_pro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Class to perform technical analysis on cryptocurrency data."""

    def __init__(self):
        """Initialize the TechnicalAnalyzer."""
        logger.info("Initializing Technical Analyzer")

    def add_basic_indicators(self, df):
        """
        Add basic technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot add indicators")
            return df

        try:
            # Check if we have a MultiIndex from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Convert MultiIndex to single level
                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

            # Make sure we have the right column names
            required_cols = ['open', 'high', 'low', 'close', 'volume']

            # Convert all column names to lowercase for comparison
            df_cols_lower = [col.lower() for col in df.columns]

            # Check if all required columns are present
            if not all(col in df_cols_lower for col in required_cols):
                logger.error(f"DataFrame missing required columns: {required_cols}")
                return df

            # Create a mapping from lowercase to actual column names
            col_mapping = {col.lower(): col for col in df.columns}

            # Create a copy with standardized column names
            df_std = df.copy()
            df_std.columns = [col.lower() for col in df.columns]

            # Add moving averages
            df_std['sma_20'] = ta.sma(df_std['close'], length=20)
            df_std['sma_50'] = ta.sma(df_std['close'], length=50)
            df_std['sma_200'] = ta.sma(df_std['close'], length=200)
            df_std['ema_12'] = ta.ema(df_std['close'], length=12)
            df_std['ema_26'] = ta.ema(df_std['close'], length=26)

            # Add Bollinger Bands
            bbands = ta.bbands(df_std['close'], length=20)
            df_std = pd.concat([df_std, bbands], axis=1)

            # Add RSI
            df_std['rsi_14'] = ta.rsi(df_std['close'], length=14)

            # Add MACD
            macd = ta.macd(df_std['close'])
            df_std = pd.concat([df_std, macd], axis=1)

            # Add Average True Range
            df_std['atr_14'] = ta.atr(df_std['high'], df_std['low'], df_std['close'], length=14)

            # Add On-Balance Volume
            df_std['obv'] = ta.obv(df_std['close'], df_std['volume'])

            logger.info("Successfully added basic indicators")
            return df_std

        except Exception as e:
            logger.error(f"Error adding basic indicators: {e}")
            return df

    def add_advanced_indicators(self, df):
        """
        Add advanced technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot add indicators")
            return df

        try:
            # Check if we have a MultiIndex from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Convert MultiIndex to single level
                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            else:
                # Standardize column names to lowercase
                df.columns = [col.lower() for col in df.columns]

            # Add Ichimoku Cloud
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df = pd.concat([df, ichimoku[0]], axis=1)  # Only use the first DataFrame

            # Add Awesome Oscillator
            df['ao'] = ta.ao(df['high'], df['low'])

            # Add Stochastic Oscillator
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df = pd.concat([df, stoch], axis=1)

            # Add Chaikin Money Flow
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])

            # Add Directional Movement Index
            adx = ta.adx(df['high'], df['low'], df['close'])
            df = pd.concat([df, adx], axis=1)

            # Add Parabolic SAR
            df['psar'] = ta.psar(df['high'], df['low'])['PSARl_0.02_0.2']

            # Add Keltner Channels
            keltner = ta.kc(df['high'], df['low'], df['close'])
            df = pd.concat([df, keltner], axis=1)

            # Add Squeeze Momentum Indicator
            squeeze = ta.squeeze(df['high'], df['low'], df['close'])
            df = pd.concat([df, squeeze], axis=1)

            logger.info("Successfully added advanced indicators")
            return df

        except Exception as e:
            logger.error(f"Error adding advanced indicators: {e}")
            return df

    def add_custom_indicators(self, df):
        """
        Add custom technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot add indicators")
            return df

        try:
            # Check if we have a MultiIndex from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Convert MultiIndex to single level
                df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
            else:
                # Standardize column names to lowercase
                df.columns = [col.lower() for col in df.columns]

            # Add Candle Patterns
            candle_patterns = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name=['doji', 'engulfing', 'hammer'])
            if not candle_patterns.empty:
                df = pd.concat([df, candle_patterns], axis=1)

            # Add Z-Score of close price
            df['zscore_14'] = ta.zscore(df['close'], length=14)

            # Add Volatility ratio
            df['volatility_ratio'] = df['atr_14'] / df['close'] * 100 if 'atr_14' in df.columns else np.nan

            # Add Price Rate of Change
            df['roc_10'] = ta.roc(df['close'], length=10)

            # Add Supertrend
            supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=14, multiplier=3.0)
            df = pd.concat([df, supertrend], axis=1)

            # Add Squeeze Pro (using our custom implementation)
            try:
                squeeze_pro_df = squeeze_pro(df['high'], df['low'], df['close'])
                if squeeze_pro_df is not None:
                    df = pd.concat([df, squeeze_pro_df], axis=1)
            except Exception as e:
                logger.warning(f"Could not calculate Squeeze Pro: {e}")

            # Add Volume Weighted Average Price (if index is datetime)
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                except Exception as e:
                    logger.warning(f"Could not calculate VWAP: {e}")

            logger.info("Successfully added custom indicators")
            return df

        except Exception as e:
            logger.error(f"Error adding custom indicators: {e}")
            return df

    def add_all_indicators(self, df):
        """
        Add all technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with all indicators
        """
        # First add basic indicators which standardizes column names
        df_with_basic = self.add_basic_indicators(df)

        if df_with_basic.empty:
            logger.warning("Basic indicators resulted in empty DataFrame")
            return df

        # Then add advanced and custom indicators
        df_with_advanced = self.add_advanced_indicators(df_with_basic)
        df_with_all = self.add_custom_indicators(df_with_advanced)

        # Check if we have any data left after adding indicators
        if df_with_all.empty:
            logger.warning("All indicators resulted in empty DataFrame")
            return df

        # Drop NaN values, but only if it doesn't make the DataFrame empty
        df_no_na = df_with_all.dropna()
        if not df_no_na.empty:
            df_with_all = df_no_na
        else:
            logger.warning("Dropping NaN values would result in empty DataFrame, keeping NaNs")
            # Fill NaN values with 0 for numeric columns to avoid issues
            numeric_cols = df_with_all.select_dtypes(include=['number']).columns
            df_with_all[numeric_cols] = df_with_all[numeric_cols].fillna(0)

        return df_with_all

    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators.

        Args:
            df (pd.DataFrame): DataFrame with technical indicators

        Returns:
            pd.DataFrame: DataFrame with added signals
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot generate signals")
            return df

        try:
            # Check if required columns exist
            required_cols = ['ema_12', 'ema_26', 'rsi_14']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing required columns for signal generation: {missing_cols}")
                # Return the original DataFrame without signals
                return df

            # Moving Average Crossover
            df['ma_crossover'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)

            # RSI Overbought/Oversold
            df['rsi_signal'] = np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0))

            # MACD Signal
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                df['macd_signal'] = np.where(df['MACD_12_26_9'] > df['MACDs_12_26_9'], 1, -1)

            # Bollinger Bands Signals
            if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0', 'close']):
                df['bb_signal'] = np.where(df['close'] < df['BBL_20_2.0'], 1,
                                          np.where(df['close'] > df['BBU_20_2.0'], -1, 0))

            # Supertrend Signal
            if 'SUPERT_14_3.0' in df.columns and 'close' in df.columns:
                df['supertrend_signal'] = np.where(df['close'] > df['SUPERT_14_3.0'], 1, -1)

            # Combined Signal (simple average of all signals)
            signal_cols = [col for col in df.columns if col.endswith('_signal') or col.endswith('_crossover')]
            if signal_cols:
                df['combined_signal'] = df[signal_cols].mean(axis=1)

                # Make signals more selective by using stricter thresholds
                df['signal'] = np.where(df['combined_signal'] > 0.6, 1,
                                       np.where(df['combined_signal'] < -0.6, -1, 0))

                # Further reduce signals by only keeping signals when they change
                # This eliminates consecutive identical signals
                df['signal_change'] = df['signal'].diff().fillna(0)
                df['signal'] = np.where(df['signal_change'] != 0, df['signal'], 0)
                df.drop('signal_change', axis=1, inplace=True)

            logger.info("Successfully generated trading signals")
            return df

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return df


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    import yfinance as yf

    # Download sample data
    data = yf.download('BTC-USD', period='1mo', interval='1d')

    # Initialize analyzer
    analyzer = TechnicalAnalyzer()

    # Add indicators
    data_with_indicators = analyzer.add_all_indicators(data)

    # Generate signals
    data_with_signals = analyzer.generate_signals(data_with_indicators)

    # Print results
    print(data_with_signals.tail())
