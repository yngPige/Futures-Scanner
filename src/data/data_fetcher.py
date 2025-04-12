"""
Data Fetcher Module for Crypto Futures Scanner

This module provides functionality to fetch cryptocurrency futures data
from various sources including CCXT-supported exchanges and yfinance.
"""

import os
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Class to fetch cryptocurrency data from various sources."""

    def __init__(self, exchange_id='binance', timeframe='1h'):
        """
        Initialize the DataFetcher.

        Args:
            exchange_id (str): The exchange ID to use (default: 'binance')
            timeframe (str): The timeframe to fetch data for (default: '1h')
        """
        self.timeframe = timeframe
        self.exchange_id = exchange_id

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures market
                }
            })
            logger.info(f"Successfully initialized {exchange_id} exchange")
        except Exception as e:
            logger.error(f"Error initializing exchange {exchange_id}: {e}")
            self.exchange = None

    def get_available_symbols(self):
        """
        Get available futures symbols from the exchange.

        Returns:
            list: List of available futures symbols
        """
        if not self.exchange:
            logger.error("Exchange not initialized")
            return []

        try:
            self.exchange.load_markets()
            symbols = [symbol for symbol in self.exchange.symbols if '/USDT' in symbol and self.exchange.markets[symbol].get('future', False)]
            logger.info(f"Found {len(symbols)} futures symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    def fetch_ohlcv(self, symbol, limit=500, since=None):
        """
        Fetch OHLCV data for a specific symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT')
            limit (int): Number of candles to fetch
            since (int, optional): Timestamp in milliseconds for start time

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()

        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since, limit)

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(self, symbols, limit=500):
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols (list): List of symbols to fetch
            limit (int): Number of candles to fetch for each symbol

        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, limit)
            if not df.empty:
                result[symbol] = df

        logger.info(f"Successfully fetched data for {len(result)} symbols")
        return result

    def fetch_from_yfinance(self, ticker, period="1y", interval="1h"):
        """
        Fetch data from Yahoo Finance as a fallback or for testing.

        Args:
            ticker (str): The ticker symbol (e.g., 'BTC-USD')
            period (str): The period to fetch (default: '1y')
            interval (str): The interval for data points (default: '1h')

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Download data from yfinance
            data = yf.download(ticker, period=period, interval=interval)
            logger.info(f"Successfully fetched {len(data)} rows from yfinance for {ticker}")

            # Standardize column names to lowercase
            if isinstance(data.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            # Ensure we have the standard OHLCV column names
            column_mapping = {
                'adj close': 'close',  # Map 'Adj Close' to 'close'
                'price': 'close',      # Map 'Price' to 'close'
            }

            # Rename columns if needed
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)

            # Check if we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                logger.warning(f"Missing columns in yfinance data: {missing_cols}")

            return data
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = DataFetcher(exchange_id='binance', timeframe='1h')

    # Get available symbols
    symbols = fetcher.get_available_symbols()
    print(f"Available symbols: {symbols[:5]}...")

    # Fetch data for BTC/USDT
    if symbols and 'BTC/USDT' in symbols:
        btc_data = fetcher.fetch_ohlcv('BTC/USDT', limit=100)
        print(btc_data.head())

    # Fetch from yfinance as fallback
    btc_yf = fetcher.fetch_from_yfinance('BTC-USD', period='1mo', interval='1d')
    print(btc_yf.head())
