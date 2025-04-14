"""
Data Fetcher Module for Crypto Futures Scanner

This module provides functionality to fetch cryptocurrency futures data
from various sources including CCXT-supported exchanges and yfinance.
"""

import pandas as pd
import ccxt
import yfinance as yf
import logging
import os
import json
import re
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "symbols")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Define pattern for complex trading pairs to filter out
COMPLEX_PAIR_PATTERN = re.compile(r'.*USDT.*\d+.*[CP]$')

class DataFetcher:
    """Class to fetch cryptocurrency data from various sources."""

    def _get_cache_path(self, exchange_id, market_type, quote_currency):
        """Get the path to the cache file for the given parameters."""
        return os.path.join(CACHE_DIR, f"{exchange_id}_{market_type}_{quote_currency}.json")

    def _is_cache_valid(self, cache_path):
        """Check if the cache file exists and is not expired."""
        if not os.path.exists(cache_path):
            return False

        # Check if the cache is expired
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        now = datetime.now()

        return (now - file_time) < self.cache_expiry

    def _read_from_cache(self, cache_path):
        """Read symbols from the cache file."""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} symbols from cache: {cache_path}")
                return data
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def _write_to_cache(self, cache_path, symbols):
        """Write symbols to the cache file."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(symbols, f)
                logger.info(f"Cached {len(symbols)} symbols to: {cache_path}")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def _filter_complex_pairs(self, symbols):
        """Filter out complex trading pairs that match the pattern."""
        filtered_symbols = [s for s in symbols if not COMPLEX_PAIR_PATTERN.match(s)]
        removed_count = len(symbols) - len(filtered_symbols)

        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} complex trading pairs")

        return filtered_symbols

    # Initialize exchange
    def __init__(self, exchange_id='kraken', timeframe='1h', market_type='spot'):
        """
        Initialize the DataFetcher.

        Args:
            exchange_id (str): The exchange ID to use (default: 'kraken')
                               Special value 'CCXT:ALL' will try all available exchanges
            timeframe (str): The timeframe to fetch data for (default: '1h')
            market_type (str): The market type to use (default: 'spot', can be 'spot' or 'future')
        """
        self.timeframe = timeframe
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.exchange = None
        self.all_exchanges = []
        self.cache_expiry = timedelta(hours=24)  # Cache expires after 24 hours

        # Special case for CCXT:ALL
        if exchange_id.upper() == 'CCXT:ALL':
            logger.info("Using CCXT:ALL mode - will try all available exchanges")
            # Store the list of available exchanges for later use
            self.all_exchanges = self.get_available_exchanges()
            return

        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': market_type,
                }
            })
            logger.info(f"Successfully initialized {exchange_id} exchange for {market_type} market")
        except Exception as e:
            logger.error(f"Error initializing exchange {exchange_id}: {e}")
            self.exchange = None

    def get_available_exchanges(self):
        """
        Get a list of available CCXT exchanges.

        Returns:
            list: List of exchange IDs that can be used with CCXT
        """
        try:
            # Get all exchange IDs from CCXT
            all_exchanges = ccxt.exchanges

            # Filter out exchanges that are known to have issues, are deprecated, or should be excluded
            blacklist = ['bitforex', 'coinex', 'bitmart', 'lbank', 'woo', 'bitget', 'binance', 'bybit']
            filtered_exchanges = [ex for ex in all_exchanges if ex not in blacklist]

            # Prioritize popular exchanges by putting them first in the list
            priority_exchanges = ['kucoin', 'okx', 'kraken', 'huobi', 'gate']

            # Sort exchanges to put priority ones first
            sorted_exchanges = []
            for ex in priority_exchanges:
                if ex in filtered_exchanges:
                    sorted_exchanges.append(ex)
                    filtered_exchanges.remove(ex)

            # Add remaining exchanges
            sorted_exchanges.extend(filtered_exchanges)

            logger.info(f"Found {len(sorted_exchanges)} available CCXT exchanges")
            return sorted_exchanges
        except Exception as e:
            logger.error(f"Error getting available exchanges: {e}")
            return []

    def get_available_symbols(self, quote_currency='USDT'):
        """
        Get available symbols from the exchange.

        Args:
            quote_currency (str): The quote currency to filter by (default: 'USDT')

        Returns:
            list: List of available symbols with the specified quote currency
        """
        # Special case for CCXT:ALL
        if self.exchange_id.upper() == 'CCXT:ALL':
            # Check if we have a valid cache for CCXT:ALL
            cache_path = self._get_cache_path('CCXT_ALL', self.market_type, quote_currency)

            if self._is_cache_valid(cache_path):
                cached_symbols = self._read_from_cache(cache_path)
                if cached_symbols is not None:
                    # Filter out complex pairs
                    filtered_symbols = self._filter_complex_pairs(cached_symbols)
                    return filtered_symbols

            # If no valid cache, fetch from exchanges
            all_symbols = set()  # Use a set to avoid duplicates

            # Try to get symbols from each exchange in our list
            for exchange_id in self.all_exchanges[:5]:  # Limit to first 5 exchanges for performance
                try:
                    # Check if we have a valid cache for this exchange
                    exchange_cache_path = self._get_cache_path(exchange_id, self.market_type, quote_currency)

                    if self._is_cache_valid(exchange_cache_path):
                        cached_symbols = self._read_from_cache(exchange_cache_path)
                        if cached_symbols is not None:
                            # Add to our set
                            all_symbols.update(cached_symbols)
                            logger.info(f"Added {len(cached_symbols)} symbols from {exchange_id} cache")
                            continue

                    # If no valid cache, fetch from exchange
                    # Create a temporary fetcher for this exchange
                    temp_fetcher = DataFetcher(exchange_id=exchange_id,
                                              timeframe=self.timeframe,
                                              market_type=self.market_type)

                    # Get symbols from this exchange
                    exchange_symbols = temp_fetcher.get_available_symbols(quote_currency)

                    # Add to our set
                    all_symbols.update(exchange_symbols)

                    logger.info(f"Added {len(exchange_symbols)} symbols from {exchange_id}")
                except Exception as e:
                    logger.error(f"Error getting symbols from {exchange_id}: {e}")

            # Convert set back to sorted list
            symbols_list = sorted(list(all_symbols))
            logger.info(f"Found total of {len(symbols_list)} unique symbols across exchanges")

            # Cache the results
            self._write_to_cache(cache_path, symbols_list)

            # Filter out complex pairs
            filtered_symbols = self._filter_complex_pairs(symbols_list)
            return filtered_symbols

        # Regular case - single exchange
        # Check if we have a valid cache
        cache_path = self._get_cache_path(self.exchange_id, self.market_type, quote_currency)

        if self._is_cache_valid(cache_path):
            cached_symbols = self._read_from_cache(cache_path)
            if cached_symbols is not None:
                # Filter out complex pairs
                filtered_symbols = self._filter_complex_pairs(cached_symbols)
                return filtered_symbols

        # If no valid cache, fetch from exchange
        if not self.exchange:
            logger.error("Exchange not initialized")
            return []

        try:
            self.exchange.load_markets()

            # Filter symbols by quote currency
            if self.market_type == 'future':
                symbols = [symbol for symbol in self.exchange.symbols
                          if f'/{quote_currency}' in symbol and
                          self.exchange.markets[symbol].get('future', False)]
                logger.info(f"Found {len(symbols)} futures symbols with {quote_currency} quote currency")
            else:
                symbols = [symbol for symbol in self.exchange.symbols
                          if f'/{quote_currency}' in symbol]
                logger.info(f"Found {len(symbols)} spot symbols with {quote_currency} quote currency")

            # Cache the results
            self._write_to_cache(cache_path, symbols)

            # Filter out complex pairs
            filtered_symbols = self._filter_complex_pairs(symbols)
            return filtered_symbols
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
        # Special case for CCXT:ALL
        if self.exchange_id.upper() == 'CCXT:ALL':
            # Try exchanges one by one until we get data
            for exchange_id in self.all_exchanges:
                try:
                    logger.info(f"Trying to fetch {symbol} from {exchange_id}...")

                    # Create a temporary fetcher for this exchange
                    temp_fetcher = DataFetcher(exchange_id=exchange_id,
                                              timeframe=self.timeframe,
                                              market_type=self.market_type)

                    # Try to fetch data
                    df = temp_fetcher.fetch_ohlcv(symbol, limit, since)

                    # If we got data, add exchange info and return
                    if not df.empty:
                        # Store the exchange ID in the DataFrame attributes
                        df.attrs['exchange'] = exchange_id
                        df.attrs['symbol'] = symbol
                        df.attrs['timeframe'] = self.timeframe

                        logger.info(f"Successfully fetched {len(df)} candles for {symbol} from {exchange_id}")
                        return df
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} from {exchange_id}: {e}")
                    continue

            # If we get here, we couldn't fetch from any exchange
            logger.error(f"Failed to fetch {symbol} from any exchange")
            return pd.DataFrame()

        # Regular case - single exchange
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()

        try:
            # Fetch OHLCV data
            try:
                # Try to fetch with current market type
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since, limit)
            except Exception as e:
                logger.warning(f"Error fetching {symbol} with market type {self.market_type}: {e}")

                # If we're in futures mode and it fails, try spot market
                if self.market_type == 'future':
                    logger.info(f"Trying to fetch {symbol} from spot market instead")
                    # Temporarily change market type
                    self.exchange.options['defaultType'] = 'spot'
                    ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since, limit)
                    # Reset market type
                    self.exchange.options['defaultType'] = self.market_type
                else:
                    # Re-raise the exception if we're already in spot mode
                    raise

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Store metadata in DataFrame attributes
            df.attrs['exchange'] = self.exchange_id
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = self.timeframe

            logger.info(f"Successfully fetched {len(df)} candles for {symbol} from {self.exchange_id}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} from {self.exchange_id}: {e}")
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
        successful_exchanges = {}

        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, limit)
            if not df.empty:
                result[symbol] = df

                # Track which exchange was used for each symbol (for CCXT:ALL mode)
                if self.exchange_id.upper() == 'CCXT:ALL' and 'exchange' in df.attrs:
                    successful_exchanges[symbol] = df.attrs['exchange']

        # Log results
        if self.exchange_id.upper() == 'CCXT:ALL' and successful_exchanges:
            exchange_summary = ', '.join([f"{symbol}: {exchange}" for symbol, exchange in successful_exchanges.items()])
            logger.info(f"Successfully fetched data using these exchanges: {exchange_summary}")

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

            # Store metadata in DataFrame attributes
            data.attrs['exchange'] = 'yfinance'
            data.attrs['symbol'] = ticker
            data.attrs['timeframe'] = interval

            return data
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Test with different exchanges and market types
    exchanges = ['kucoin', 'huobi', 'kraken']

    for exchange_id in exchanges:
        print(f"\n{'=' * 50}")
        print(f"Testing {exchange_id.upper()} exchange")
        print(f"{'=' * 50}")

        # Test spot market
        print("\nSPOT MARKET:")
        spot_fetcher = DataFetcher(exchange_id=exchange_id, timeframe='1h', market_type='spot')
        spot_symbols = spot_fetcher.get_available_symbols(quote_currency='USDT')
        print(f"Available USDT spot pairs: {len(spot_symbols)}")
        if spot_symbols:
            print(f"Sample symbols: {spot_symbols[:5]}...")

            # Fetch data for BTC/USDT if available
            if 'BTC/USDT' in spot_symbols:
                print("\nFetching BTC/USDT data...")
                btc_data = spot_fetcher.fetch_ohlcv('BTC/USDT', limit=10)
                if not btc_data.empty:
                    print(btc_data.head(3))

        # Test futures market
        print("\nFUTURES MARKET:")
        futures_fetcher = DataFetcher(exchange_id=exchange_id, timeframe='1h', market_type='future')
        futures_symbols = futures_fetcher.get_available_symbols(quote_currency='USDT')
        print(f"Available USDT futures pairs: {len(futures_symbols)}")
        if futures_symbols:
            print(f"Sample symbols: {futures_symbols[:5]}...")

            # Fetch data for BTC/USDT if available
            if 'BTC/USDT' in futures_symbols:
                print("\nFetching BTC/USDT futures data...")
                btc_futures_data = futures_fetcher.fetch_ohlcv('BTC/USDT', limit=10)
                if not btc_futures_data.empty:
                    print(btc_futures_data.head(3))

    # Fetch from yfinance as fallback
    print("\n\nYFINANCE FALLBACK:")
    yf_fetcher = DataFetcher(exchange_id='kraken', timeframe='1d')
    btc_yf = yf_fetcher.fetch_from_yfinance('BTC-USD', period='1mo', interval='1d')
    print(btc_yf.head(3))
