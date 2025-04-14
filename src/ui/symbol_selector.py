"""
Symbol Selector Module for Crypto Futures Scanner

This module provides a popup window for selecting cryptocurrency symbols
with a more user-friendly interface and symbol availability checking.
"""

import os
import time
import logging
import threading
from colorama import Fore, Style, init
from src.data.data_fetcher import DataFetcher

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_loading_animation(duration=3):
    """Show a simple loading bar animation for the specified duration."""
    start_time = time.time()
    bar_width = 50

    while time.time() - start_time < duration:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Calculate progress based on elapsed time
        progress = min(1.0, (time.time() - start_time) / duration)
        filled_width = int(bar_width * progress)

        # Create the loading bar
        bar = f"[{Fore.GREEN}{'█' * filled_width}{Style.RESET_ALL}{' ' * (bar_width - filled_width)}] {int(progress * 100)}%"

        # Print loading message and bar
        print(f"\n\n{Fore.CYAN}Loading symbols...{Style.RESET_ALL}\n")
        print(bar + "\n")

        # Sleep briefly
        time.sleep(0.1)

    # Clear screen one last time
    os.system('cls' if os.name == 'nt' else 'clear')

# Check if curses is available
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    logger.warning("Curses module not available. Using fallback terminal UI for symbol selector.")

class SymbolSelector:
    """Class to provide a popup window for selecting cryptocurrency symbols."""

    def __init__(self, current_symbol='BTC/USDT', current_exchange='kraken'):
        """
        Initialize the SymbolSelector.

        Args:
            current_symbol (str): The currently selected symbol
            current_exchange (str): The currently selected exchange
        """
        self.current_symbol = current_symbol
        self.current_exchange = current_exchange
        self.search_term = ""
        self.selected_index = 0
        self.scroll_offset = 0
        self.max_display_rows = 15
        self.symbols_by_category = {}
        self.all_symbols = []
        self.filtered_symbols = []
        self.categories = []
        self.exchange_symbols = {}
        self.available_exchanges = ['CCXT:ALL', 'kraken', 'kucoin', 'huobi']

        # Base cryptocurrencies for generating symbols
        self.base_cryptocurrencies = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC',
            'LINK', 'UNI', 'LTC', 'SHIB', 'TRX', 'ETC', 'TON', 'NEAR', 'BCH', 'ATOM'
        ]

        # Quote currencies and formats
        self.quote_formats = [
            {'quote': 'USDT', 'format': '{}/{}'},  # Format for USDT pairs (e.g., BTC/USDT)
            {'quote': 'USD', 'format': '{}-{}'}    # Format for USD pairs (e.g., BTC-USD)
        ]

        # Load symbols
        self._load_symbols()

    def _load_symbols(self):
        """Load symbols from exchanges and organize them by category."""
        # Generate available symbols
        self.all_symbols = []

        # First, add common symbols from base cryptocurrencies
        for base in self.base_cryptocurrencies:
            for quote_format in self.quote_formats:
                symbol = quote_format['format'].format(base, quote_format['quote'])
                self.all_symbols.append(symbol)

        # Organize by quote currency
        self.symbols_by_category = {}
        for quote_format in self.quote_formats:
            quote = quote_format['quote']
            if quote == 'USD':
                symbols = [s for s in self.all_symbols if s.endswith(f'-{quote}')]
            else:
                symbols = [s for s in self.all_symbols if s.endswith(f'/{quote}')]

            self.symbols_by_category[f"{quote} pairs"] = sorted(symbols)

        # Set categories
        self.categories = list(self.symbols_by_category.keys())

        # Set filtered symbols to all symbols initially
        self.filtered_symbols = self.all_symbols.copy()

        # Try to find the current symbol in the list
        try:
            self.selected_index = self.filtered_symbols.index(self.current_symbol)
        except ValueError:
            self.selected_index = 0

        # Load exchange symbols in background
        self._load_exchange_symbols()

    def _load_exchange_symbols(self):
        """Load available symbols from exchanges."""
        # Start loading animation in a separate thread
        loading_thread = threading.Thread(target=show_loading_animation, args=(5,))
        loading_thread.daemon = True
        loading_thread.start()

        # Temporarily disable INFO logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        try:
            for exchange_id in self.available_exchanges:
                try:
                    # Try to get both spot and futures symbols
                    spot_fetcher = DataFetcher(exchange_id=exchange_id, market_type='spot')
                    futures_fetcher = DataFetcher(exchange_id=exchange_id, market_type='future')

                    # Get USDT pairs from both markets
                    spot_symbols = spot_fetcher.get_available_symbols(quote_currency='USDT')
                    futures_symbols = futures_fetcher.get_available_symbols(quote_currency='USDT')

                    # Combine symbols (remove duplicates)
                    all_symbols = list(set(spot_symbols + futures_symbols))

                    self.exchange_symbols[exchange_id] = all_symbols
                    # Don't log info messages
                except Exception as e:
                    # Only log errors
                    logger.error(f"Error loading symbols from {exchange_id}: {e}")
                    self.exchange_symbols[exchange_id] = []
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

            # Wait for the loading animation to finish
            if loading_thread.is_alive():
                loading_thread.join(timeout=0.5)

    def _filter_symbols(self):
        """Filter symbols based on search term."""
        if not self.search_term:
            self.filtered_symbols = self.all_symbols.copy()
        else:
            search_term = self.search_term.upper()
            self.filtered_symbols = [s for s in self.all_symbols if search_term in s]

        # Reset selection and scroll
        self.selected_index = 0 if self.filtered_symbols else -1
        self.scroll_offset = 0

    def _check_symbol_availability(self, symbol, exchange):
        """
        Check if a symbol is available on the specified exchange.

        Args:
            symbol (str): The symbol to check
            exchange (str): The exchange to check

        Returns:
            bool: True if the symbol is available, False otherwise
        """
        # If we haven't loaded symbols for this exchange yet, assume it's available
        if exchange not in self.exchange_symbols:
            return True

        return symbol in self.exchange_symbols[exchange]

    def _find_alternative_symbol(self, symbol, exchange):
        """
        Find an alternative symbol on the specified exchange.

        Args:
            symbol (str): The symbol that's not available
            exchange (str): The exchange to check

        Returns:
            str or None: An alternative symbol if found, None otherwise
        """
        # If we haven't loaded symbols for this exchange, return None
        if exchange not in self.exchange_symbols:
            return None

        # Try to find the base currency
        if '/' in symbol:
            base = symbol.split('/')[0]
        elif '-' in symbol:
            base = symbol.split('-')[0]
        else:
            return None

        # Look for any symbol with the same base currency
        for s in self.exchange_symbols[exchange]:
            if s.startswith(f"{base}/") or s.startswith(f"{base}-"):
                return s

        return None

    def run(self, stdscr):
        """
        Run the symbol selector in a curses window.

        Args:
            stdscr: The curses window

        Returns:
            tuple: (selected_symbol, exchange, cancelled)
                selected_symbol (str): The selected symbol
                exchange (str): The selected exchange
                cancelled (bool): True if the selection was cancelled
        """
        # Set up curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)

        # Get screen dimensions
        max_y, max_x = stdscr.getmaxyx()

        # Calculate window dimensions
        win_height = min(25, max_y - 4)
        win_width = min(60, max_x - 4)

        # Create window centered on screen
        win_y = (max_y - win_height) // 2
        win_x = (max_x - win_width) // 2

        # Create window
        win = curses.newwin(win_height, win_width, win_y, win_x)
        win.keypad(True)

        # Main loop
        cancelled = False
        while True:
            win.clear()

            # Draw border
            win.box()

            # Draw title
            title = " Symbol Selector "
            win.addstr(0, (win_width - len(title)) // 2, title, curses.A_BOLD)

            # Draw search box
            search_prompt = "Search: "
            win.addstr(2, 2, search_prompt)
            win.addstr(2, 2 + len(search_prompt), self.search_term)

            # Draw current exchange
            exchange_prompt = "Exchange: "
            win.addstr(2, win_width - 20, exchange_prompt)
            win.addstr(2, win_width - 20 + len(exchange_prompt), self.current_exchange, curses.color_pair(2))

            # Draw symbols
            list_y_start = 4
            list_height = win_height - 7

            if not self.filtered_symbols:
                win.addstr(list_y_start + list_height // 2, (win_width - 20) // 2, "No symbols found", curses.color_pair(4))
            else:
                # Adjust scroll offset if needed
                if self.selected_index < self.scroll_offset:
                    self.scroll_offset = self.selected_index
                elif self.selected_index >= self.scroll_offset + list_height:
                    self.scroll_offset = self.selected_index - list_height + 1

                # Draw visible symbols
                visible_symbols = self.filtered_symbols[self.scroll_offset:self.scroll_offset + list_height]
                for i, symbol in enumerate(visible_symbols):
                    y = list_y_start + i

                    # Check if this symbol is available on the current exchange
                    is_available = self._check_symbol_availability(symbol, self.current_exchange)

                    # Highlight selected symbol
                    if i + self.scroll_offset == self.selected_index:
                        attr = curses.A_REVERSE
                    else:
                        attr = curses.A_NORMAL

                    # Use different color for available/unavailable symbols
                    if is_available:
                        color = curses.color_pair(1)  # Green for available
                    else:
                        color = curses.color_pair(4)  # Red for unavailable

                    win.addstr(y, 2, symbol, attr | color)

            # Draw scrollbar if needed
            if len(self.filtered_symbols) > list_height:
                scrollbar_height = max(1, int(list_height * list_height / len(self.filtered_symbols)))
                scrollbar_pos = min(list_height - scrollbar_height, int(self.scroll_offset * list_height / len(self.filtered_symbols)))

                for i in range(list_height):
                    if i >= scrollbar_pos and i < scrollbar_pos + scrollbar_height:
                        win.addch(list_y_start + i, win_width - 2, curses.ACS_BLOCK)
                    else:
                        win.addch(list_y_start + i, win_width - 2, curses.ACS_VLINE)

            # Draw instructions
            win.addstr(win_height - 2, 2, "Enter: Select  Tab: Change Exchange  Esc: Cancel", curses.color_pair(3))

            # Refresh window
            win.refresh()

            # Get input
            key = win.getch()

            # Handle input
            if key == curses.KEY_UP:
                self.selected_index = max(0, self.selected_index - 1)
            elif key == curses.KEY_DOWN:
                self.selected_index = min(len(self.filtered_symbols) - 1, self.selected_index + 1)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.selected_index = max(0, self.selected_index - list_height)
            elif key == curses.KEY_NPAGE:  # Page Down
                self.selected_index = min(len(self.filtered_symbols) - 1, self.selected_index + list_height)
            elif key == curses.KEY_HOME:
                self.selected_index = 0
            elif key == curses.KEY_END:
                self.selected_index = len(self.filtered_symbols) - 1
            elif key == 9:  # Tab key
                # Cycle through exchanges
                current_index = self.available_exchanges.index(self.current_exchange)
                next_index = (current_index + 1) % len(self.available_exchanges)
                self.current_exchange = self.available_exchanges[next_index]
            elif key == 27:  # Escape key
                cancelled = True
                break
            elif key == 10 or key == 13:  # Enter key
                if self.selected_index >= 0 and self.selected_index < len(self.filtered_symbols):
                    selected_symbol = self.filtered_symbols[self.selected_index]

                    # Check if the symbol is available on the selected exchange
                    if not self._check_symbol_availability(selected_symbol, self.current_exchange):
                        # Find an alternative symbol
                        alternative = self._find_alternative_symbol(selected_symbol, self.current_exchange)

                        # Show a message about the alternative
                        win.clear()
                        win.box()
                        win.addstr(0, (win_width - len(title)) // 2, title, curses.A_BOLD)

                        msg1 = f"{selected_symbol} is not available on {self.current_exchange}"
                        win.addstr(win_height // 2 - 2, (win_width - len(msg1)) // 2, msg1, curses.color_pair(4))

                        if alternative:
                            msg2 = f"Would you like to use {alternative} instead?"
                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(2))

                            msg3 = "Y: Yes  N: No  Tab: Change Exchange"
                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))

                            win.refresh()

                            # Get response
                            while True:
                                key = win.getch()
                                if key in [ord('y'), ord('Y')]:
                                    selected_symbol = alternative
                                    break
                                elif key in [ord('n'), ord('N')]:
                                    # Go back to selection
                                    break
                                elif key == 9:  # Tab key
                                    # Cycle through exchanges
                                    current_index = self.available_exchanges.index(self.current_exchange)
                                    next_index = (current_index + 1) % len(self.available_exchanges)
                                    self.current_exchange = self.available_exchanges[next_index]

                                    # Update message
                                    win.clear()
                                    win.box()
                                    win.addstr(0, (win_width - len(title)) // 2, title, curses.A_BOLD)

                                    msg1 = f"{selected_symbol} is not available on {self.current_exchange}"
                                    win.addstr(win_height // 2 - 2, (win_width - len(msg1)) // 2, msg1, curses.color_pair(4))

                                    # Check if the symbol is available on the new exchange
                                    if self._check_symbol_availability(selected_symbol, self.current_exchange):
                                        msg2 = f"{selected_symbol} is available on {self.current_exchange}!"
                                        win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(1))

                                        msg3 = "Enter: Select  Tab: Change Exchange  Esc: Cancel"
                                        win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))
                                    else:
                                        # Find a new alternative
                                        alternative = self._find_alternative_symbol(selected_symbol, self.current_exchange)

                                        if alternative:
                                            msg2 = f"Would you like to use {alternative} instead?"
                                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(2))

                                            msg3 = "Y: Yes  N: No  Tab: Change Exchange"
                                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))
                                        else:
                                            msg2 = f"No alternative found on {self.current_exchange}"
                                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(4))

                                            msg3 = "Tab: Change Exchange  Esc: Cancel"
                                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))

                                    win.refresh()
                                elif key == 27:  # Escape key
                                    # Go back to selection
                                    break
                                elif key == 10 or key == 13:  # Enter key
                                    # If the symbol is available on the current exchange, select it
                                    if self._check_symbol_availability(selected_symbol, self.current_exchange):
                                        break

                            if key in [ord('n'), ord('N'), 27]:  # N or Escape
                                continue
                        else:
                            msg2 = f"No alternative found on {self.current_exchange}"
                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(4))

                            msg3 = "Tab: Change Exchange  Esc: Cancel"
                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))

                            win.refresh()

                            # Get response
                            while True:
                                key = win.getch()
                                if key == 9:  # Tab key
                                    # Cycle through exchanges
                                    current_index = self.available_exchanges.index(self.current_exchange)
                                    next_index = (current_index + 1) % len(self.available_exchanges)
                                    self.current_exchange = self.available_exchanges[next_index]

                                    # Update message
                                    win.clear()
                                    win.box()
                                    win.addstr(0, (win_width - len(title)) // 2, title, curses.A_BOLD)

                                    msg1 = f"{selected_symbol} is not available on {self.current_exchange}"
                                    win.addstr(win_height // 2 - 2, (win_width - len(msg1)) // 2, msg1, curses.color_pair(4))

                                    # Check if the symbol is available on the new exchange
                                    if self._check_symbol_availability(selected_symbol, self.current_exchange):
                                        msg2 = f"{selected_symbol} is available on {self.current_exchange}!"
                                        win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(1))

                                        msg3 = "Enter: Select  Tab: Change Exchange  Esc: Cancel"
                                        win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))
                                    else:
                                        # Find a new alternative
                                        alternative = self._find_alternative_symbol(selected_symbol, self.current_exchange)

                                        if alternative:
                                            msg2 = f"Would you like to use {alternative} instead?"
                                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(2))

                                            msg3 = "Y: Yes  N: No  Tab: Change Exchange"
                                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))
                                        else:
                                            msg2 = f"No alternative found on {self.current_exchange}"
                                            win.addstr(win_height // 2, (win_width - len(msg2)) // 2, msg2, curses.color_pair(4))

                                            msg3 = "Tab: Change Exchange  Esc: Cancel"
                                            win.addstr(win_height // 2 + 2, (win_width - len(msg3)) // 2, msg3, curses.color_pair(3))

                                    win.refresh()
                                elif key == 27:  # Escape key
                                    # Go back to selection
                                    break
                                elif key == 10 or key == 13:  # Enter key
                                    # If the symbol is available on the current exchange, select it
                                    if self._check_symbol_availability(selected_symbol, self.current_exchange):
                                        break

                            if key == 27:  # Escape
                                continue

                    break
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:  # Backspace key
                if self.search_term:
                    self.search_term = self.search_term[:-1]
                    self._filter_symbols()
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.search_term += chr(key)
                self._filter_symbols()

        # Return selected symbol and exchange
        if cancelled or self.selected_index < 0 or self.selected_index >= len(self.filtered_symbols):
            return None, self.current_exchange, True
        else:
            return self.filtered_symbols[self.selected_index], self.current_exchange, False

def select_symbol_fallback(current_symbol='BTC/USDT', current_exchange='coinbase'):
    """
    Show a terminal-based UI for selecting a cryptocurrency symbol (fallback version).

    Args:
        current_symbol (str): The currently selected symbol
        current_exchange (str): The currently selected exchange

    Returns:
        tuple: (selected_symbol, exchange, cancelled)
            selected_symbol (str): The selected symbol
            exchange (str): The selected exchange
            cancelled (bool): True if the selection was cancelled
    """
    selector = SymbolSelector(current_symbol, current_exchange)

    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{' ' * 30}Symbol Selector{' ' * 30}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    # Load exchange symbols with simple loading bar
    # Start loading animation in a separate thread
    loading_thread = threading.Thread(target=show_loading_animation, args=(5,))
    loading_thread.daemon = True
    loading_thread.start()

    try:
        selector._load_exchange_symbols()
    finally:
        # Wait for the loading animation to finish
        if loading_thread.is_alive():
            loading_thread.join(timeout=0.5)

    # Display available symbols by category
    categories = list(selector.symbols_by_category.keys())

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{' ' * 30}Symbol Selector{' ' * 30}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

        print(f"Current Exchange: {Fore.GREEN}{current_exchange}{Style.RESET_ALL}")
        print(f"Current Symbol: {Fore.GREEN}{current_symbol}{Style.RESET_ALL}\n")

        print("Available Categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")

        print("\nOptions:")
        print(f"s. Search for a symbol")
        print(f"e. Change exchange")
        print(f"c. Check symbol availability")
        print(f"b. Back (cancel)")

        choice = input("\nEnter your choice: ")

        if choice.lower() == 'b':
            return current_symbol, current_exchange, True
        elif choice.lower() == 'e':
            # Change exchange
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{' ' * 30}Change Exchange{' ' * 30}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

            print("Available Exchanges:")
            for i, exchange in enumerate(selector.available_exchanges, 1):
                print(f"{i}. {exchange}")

            exchange_choice = input("\nEnter exchange number or name (or 'b' to go back): ")

            if exchange_choice.lower() == 'b':
                continue

            try:
                index = int(exchange_choice) - 1
                if 0 <= index < len(selector.available_exchanges):
                    current_exchange = selector.available_exchanges[index]
                    # Reload symbols for the new exchange
                    selector.current_exchange = current_exchange
                    # Start loading animation in a separate thread
                    loading_thread = threading.Thread(target=show_loading_animation, args=(5,))
                    loading_thread.daemon = True
                    loading_thread.start()

                    try:
                        selector._load_exchange_symbols()
                    finally:
                        # Wait for the loading animation to finish
                        if loading_thread.is_alive():
                            loading_thread.join(timeout=0.5)
            except ValueError:
                # Custom exchange name
                if exchange_choice.lower() in [e.lower() for e in selector.available_exchanges]:
                    # Find the correct case for the exchange name
                    for e in selector.available_exchanges:
                        if e.lower() == exchange_choice.lower():
                            current_exchange = e
                            # Reload symbols for the new exchange
                            selector.current_exchange = current_exchange
                            # Start loading animation in a separate thread
                            loading_thread = threading.Thread(target=show_loading_animation, args=(5,))
                            loading_thread.daemon = True
                            loading_thread.start()

                            try:
                                selector._load_exchange_symbols()
                            finally:
                                # Wait for the loading animation to finish
                                if loading_thread.is_alive():
                                    loading_thread.join(timeout=0.5)
                            break
                else:
                    print(f"\n{Fore.RED}Exchange '{exchange_choice}' not recognized. Please try again.{Style.RESET_ALL}")
                    time.sleep(2)
        elif choice.lower() == 's':
            # Search for a symbol
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{' ' * 30}Search Symbol{' ' * 30}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

            search_term = input("Enter search term (or 'b' to go back): ")

            if search_term.lower() == 'b':
                continue

            # Filter symbols based on search term
            search_term = search_term.upper()
            filtered_symbols = [s for s in selector.all_symbols if search_term in s]

            if not filtered_symbols:
                print(f"\n{Fore.RED}No symbols found matching '{search_term}'. Please try again.{Style.RESET_ALL}")
                time.sleep(2)
                continue

            # Display filtered symbols
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{' ' * 30}Search Results{' ' * 30}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

            print(f"Search results for '{search_term}':")
            for i, symbol in enumerate(filtered_symbols, 1):
                # Check if this symbol is available on the current exchange
                is_available = selector._check_symbol_availability(symbol, current_exchange)
                if is_available:
                    print(f"{i}. {Fore.GREEN}{symbol}{Style.RESET_ALL}")
                else:
                    print(f"{i}. {Fore.RED}{symbol}{Style.RESET_ALL} (not available on {current_exchange})")

            symbol_choice = input("\nEnter symbol number (or 'b' to go back): ")

            if symbol_choice.lower() == 'b':
                continue

            try:
                index = int(symbol_choice) - 1
                if 0 <= index < len(filtered_symbols):
                    selected_symbol = filtered_symbols[index]

                    # Check if the symbol is available on the selected exchange
                    if not selector._check_symbol_availability(selected_symbol, current_exchange):
                        # Find an alternative symbol
                        alternative = selector._find_alternative_symbol(selected_symbol, current_exchange)

                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}{' ' * 30}Symbol Not Available{' ' * 30}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

                        print(f"{Fore.RED}{selected_symbol} is not available on {current_exchange}{Style.RESET_ALL}")

                        if alternative:
                            print(f"\n{Fore.GREEN}Would you like to use {alternative} instead?{Style.RESET_ALL}")
                            alt_choice = input("\nEnter 'y' to use alternative, 'n' to go back: ")

                            if alt_choice.lower() == 'y':
                                return alternative, current_exchange, False
                        else:
                            print(f"\n{Fore.RED}No alternative found on {current_exchange}{Style.RESET_ALL}")
                            print("\nYou can try changing the exchange or selecting a different symbol.")
                            input("\nPress Enter to continue...")
                    else:
                        return selected_symbol, current_exchange, False
                else:
                    print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                    time.sleep(2)
            except ValueError:
                print(f"\n{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
                time.sleep(2)
        elif choice.lower() == 'c':
            # Check symbol availability
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{' ' * 30}Check Symbol{' ' * 30}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

            symbol_to_check = input("Enter symbol to check (or 'b' to go back): ")

            if symbol_to_check.lower() == 'b':
                continue

            # Check if the symbol is available on the current exchange
            is_available = selector._check_symbol_availability(symbol_to_check, current_exchange)

            if is_available:
                print(f"\n{Fore.GREEN}{symbol_to_check} is available on {current_exchange}{Style.RESET_ALL}")
                use_choice = input("\nUse this symbol? (y/n): ")

                if use_choice.lower() == 'y':
                    return symbol_to_check, current_exchange, False
            else:
                print(f"\n{Fore.RED}{symbol_to_check} is not available on {current_exchange}{Style.RESET_ALL}")

                # Find an alternative symbol
                alternative = selector._find_alternative_symbol(symbol_to_check, current_exchange)

                if alternative:
                    print(f"\n{Fore.GREEN}Alternative: {alternative}{Style.RESET_ALL}")
                    alt_choice = input("\nUse this alternative? (y/n): ")

                    if alt_choice.lower() == 'y':
                        return alternative, current_exchange, False
                else:
                    print(f"\n{Fore.RED}No alternative found on {current_exchange}{Style.RESET_ALL}")

            input("\nPress Enter to continue...")
        elif choice.isdigit():
            # Select a category
            try:
                index = int(choice) - 1
                if 0 <= index < len(categories):
                    category = categories[index]
                    symbols = selector.symbols_by_category[category]

                    # Display symbols in this category
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{' ' * 30}{category}{' ' * 30}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

                    for i, symbol in enumerate(symbols, 1):
                        # Check if this symbol is available on the current exchange
                        is_available = selector._check_symbol_availability(symbol, current_exchange)
                        if is_available:
                            print(f"{i}. {Fore.GREEN}{symbol}{Style.RESET_ALL}")
                        else:
                            print(f"{i}. {Fore.RED}{symbol}{Style.RESET_ALL} (not available on {current_exchange})")

                    symbol_choice = input("\nEnter symbol number (or 'b' to go back): ")

                    if symbol_choice.lower() == 'b':
                        continue

                    try:
                        index = int(symbol_choice) - 1
                        if 0 <= index < len(symbols):
                            selected_symbol = symbols[index]

                            # Check if the symbol is available on the selected exchange
                            if not selector._check_symbol_availability(selected_symbol, current_exchange):
                                # Find an alternative symbol
                                alternative = selector._find_alternative_symbol(selected_symbol, current_exchange)

                                os.system('cls' if os.name == 'nt' else 'clear')
                                print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
                                print(f"{Fore.CYAN}{' ' * 30}Symbol Not Available{' ' * 30}{Style.RESET_ALL}")
                                print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

                                print(f"{Fore.RED}{selected_symbol} is not available on {current_exchange}{Style.RESET_ALL}")

                                if alternative:
                                    print(f"\n{Fore.GREEN}Would you like to use {alternative} instead?{Style.RESET_ALL}")
                                    alt_choice = input("\nEnter 'y' to use alternative, 'n' to go back: ")

                                    if alt_choice.lower() == 'y':
                                        return alternative, current_exchange, False
                                else:
                                    print(f"\n{Fore.RED}No alternative found on {current_exchange}{Style.RESET_ALL}")
                                    print("\nYou can try changing the exchange or selecting a different symbol.")
                                    input("\nPress Enter to continue...")
                            else:
                                return selected_symbol, current_exchange, False
                        else:
                            print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                            time.sleep(2)
                    except ValueError:
                        print(f"\n{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
                        time.sleep(2)
                else:
                    print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                    time.sleep(2)
            except ValueError:
                print(f"\n{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")
                time.sleep(2)
        else:
            print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
            time.sleep(2)

def select_symbol(current_symbol='BTC/USDT', current_exchange='kraken'):
    """
    Show a popup window for selecting a cryptocurrency symbol.

    Args:
        current_symbol (str): The currently selected symbol
        current_exchange (str): The currently selected exchange

    Returns:
        tuple: (selected_symbol, exchange, cancelled)
            selected_symbol (str): The selected symbol
            exchange (str): The selected exchange
            cancelled (bool): True if the selection was cancelled
    """
    # Try to use the Tkinter selector first
    try:
        from src.ui.tkinter_symbol_selector import select_symbol_tkinter
        return select_symbol_tkinter(current_symbol, current_exchange)
    except Exception as e:
        print(f"Error using Tkinter selector: {e}")

        # Fall back to curses if available
        if CURSES_AVAILABLE:
            selector = SymbolSelector(current_symbol, current_exchange)

            # Initialize curses
            result = curses.wrapper(selector.run)

            # Clean up terminal
            os.system('cls' if os.name == 'nt' else 'clear')

            return result
        else:
            # Use fallback version
            return select_symbol_fallback(current_symbol, current_exchange)


# Example usage
if __name__ == "__main__":
    symbol, exchange, cancelled = select_symbol()

    if cancelled:
        print("Selection cancelled")
    else:
        print(f"Selected symbol: {symbol} on {exchange}")
