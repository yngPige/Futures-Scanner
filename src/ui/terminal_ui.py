"""
Terminal UI Module for Crypto Futures Scanner

This module provides a terminal-based user interface for the Crypto Futures Scanner application.
"""

import os
import time
import logging
import argparse
from datetime import datetime
from colorama import init, Fore, Back, Style
from src.ui.terminal_output import TerminalOutputGenerator

# Initialize colorama
init()

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




class TerminalUI:
    """Terminal-based user interface for Crypto Futures Scanner."""

    def __init__(self):
        """Initialize the Terminal UI."""
        self.running = True
        self.current_menu = self.main_menu

        # Default settings
        self.settings = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'limit': 500,
            'exchange': 'coinbase',
            'model_type': 'random_forest',
            'model_path': None,
            'theme': 'dark',
            'save': True,
            'tune': False,
            'use_llm': False,
            'llm_model': 'llama3-8b',
            'use_gpu': False
        }

        # Track completed functions
        self.completed_functions = {
            'fetch_data': False,
            'analyze_data': False,
            'train_model': False,
            'make_predictions': False,
            'backtest_strategy': False,
            'llm_analysis': False
        }

        # Create terminal output generator
        self.output_generator = TerminalOutputGenerator(theme=self.settings['theme'])

        # Available options
        # Define base cryptocurrencies
        self.base_cryptocurrencies = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC'
        ]

        # Define quote currencies and formats
        self.quote_formats = [
            {'quote': 'USDT', 'format': '{}/{}'},  # Format for USDT pairs (e.g., BTC/USDT)
            {'quote': 'USD', 'format': '{}-{}'}    # Format for USD pairs (e.g., BTC-USD)
        ]

        # Generate available symbols
        self.available_symbols = []
        for base in self.base_cryptocurrencies:
            for quote_format in self.quote_formats:
                symbol = quote_format['format'].format(base, quote_format['quote'])
                self.available_symbols.append(symbol)

        self.available_timeframes = [
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        ]

        self.available_model_types = [
            'random_forest', 'gradient_boosting'
        ]

        self.available_themes = [
            'dark', 'light'
        ]

        # Load available models
        self.refresh_available_models()

    def refresh_available_models(self):
        """Refresh the list of available models."""
        self.available_models = []
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.joblib') and not file.endswith('_scaler.joblib') and not file.endswith('_features.joblib'):
                    self.available_models.append(os.path.join('models', file))

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title):
        """Print a header with the given title."""
        self.clear_screen()
        width = 80
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()

    def print_menu_item(self, key, description):
        """Print a menu item with the given key and description."""
        print(f"{Fore.GREEN}{key}{Style.RESET_ALL}: {description}")

    def print_settings(self, inline=False):
        """Print the current settings.

        Args:
            inline (bool): If True, print settings in a compact format for inline display.
        """
        if inline:
            settings_str = ", ".join([f"{key}: {value}" for key, value in self.settings.items() if value is not None])
            print(Fore.YELLOW + f"Settings: {settings_str}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "\nCurrent Settings:" + Style.RESET_ALL)
            for key, value in self.settings.items():
                if value is not None:
                    print(f"  {key}: {value}")
            print()

    def get_input(self, prompt):
        """Get input from the user with the given prompt."""
        return input(Fore.GREEN + prompt + Style.RESET_ALL)

    def print_success(self, message):
        """Print a success message."""
        print(Fore.GREEN + message + Style.RESET_ALL)

    def print_error(self, message):
        """Print an error message."""
        print(Fore.RED + message + Style.RESET_ALL)

    def print_info(self, message):
        """Print an info message."""
        print(Fore.BLUE + message + Style.RESET_ALL)

    def print_warning(self, message):
        """Print a warning message."""
        print(Fore.YELLOW + message + Style.RESET_ALL)

    def wait_for_key(self):
        """Wait for the user to press a key."""
        input(Fore.GREEN + "\nPress Enter to continue..." + Style.RESET_ALL)

    def show_loading_animation(self, message, duration=3, width=50, log_messages=None):
        """Show an enhanced loading bar animation with cool logs.

        Args:
            message (str): Message to display before the loading bar
            duration (int): Duration of the animation in seconds
            width (int): Width of the loading bar
            log_messages (list, optional): List of log messages to display during animation
        """
        # Initialize collected_logs if it doesn't exist
        if not hasattr(self, 'collected_logs'):
            self.collected_logs = []

        # Print header with message
        print(f"\n{Fore.CYAN}⚡ {message} ⚡{Style.RESET_ALL}")

        # Calculate steps based on duration
        steps = 20  # Total animation steps
        sleep_time = duration / steps

        # Generate random log messages if none provided
        if not log_messages:
            log_messages = [
                "Initializing data structures...",
                "Analyzing market patterns...",
                "Processing technical indicators...",
                "Calculating momentum signals...",
                "Evaluating trend strength...",
                "Checking support/resistance levels...",
                "Validating data integrity...",
                "Optimizing analysis parameters...",
                "Applying machine learning models...",
                "Finalizing results..."
            ]

        # Create a list to store displayed logs for this animation
        displayed_logs = []
        log_display_height = 5  # Number of log lines to show at once
        log_display_area = [''] * log_display_height

        # Print initial empty lines for the log display area
        for _ in range(log_display_height):
            print()

        # Show loading bar animation with logs
        for i in range(steps + 1):
            # Calculate progress percentage
            percent = i * 100 // steps

            # Calculate number of filled blocks
            filled_blocks = i * width // steps
            empty_blocks = width - filled_blocks

            # Create a gradient effect for the loading bar
            gradient_bar = ''
            for j in range(filled_blocks):
                # Calculate position in the gradient (0 to 1)
                pos = j / width if width > 0 else 0

                # Create a smooth color transition from red to yellow to green
                if pos < 0.3:
                    # Red to Yellow transition
                    color = Fore.RED
                elif pos < 0.6:
                    # Yellow to Green transition
                    color = Fore.YELLOW
                else:
                    # Green
                    color = Fore.GREEN

                gradient_bar += color + '█' + Style.RESET_ALL

            # Create the loading bar with semi-transparent background
            bar_bg = Fore.BLACK + Back.WHITE
            progress_text = f" {percent}% "
            bar = f"{bar_bg}[{Style.RESET_ALL}{gradient_bar}{' ' * empty_blocks}{bar_bg}]{Style.RESET_ALL} {progress_text}"

            # Display a log message at certain intervals
            if i > 0 and i % (steps // (len(log_messages) - 1) or 1) == 0 and log_messages:
                log_idx = min(i // (steps // (len(log_messages) - 1) or 1), len(log_messages) - 1)
                log_msg = log_messages[log_idx]

                # Choose a color for the log message based on content
                if "error" in log_msg.lower() or "failed" in log_msg.lower():
                    log_color = Fore.RED
                elif "warning" in log_msg.lower():
                    log_color = Fore.YELLOW
                elif "completed" in log_msg.lower() or "success" in log_msg.lower():
                    log_color = Fore.GREEN
                else:
                    # Cycle through colors for regular messages
                    log_colors = [Fore.CYAN, Fore.MAGENTA, Fore.BLUE, Fore.GREEN, Fore.YELLOW]
                    log_color = log_colors[log_idx % len(log_colors)]

                # Format timestamp
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                # Create formatted log message
                formatted_log = f"{Fore.WHITE}[{timestamp}]{Style.RESET_ALL} {log_color}{log_msg}{Style.RESET_ALL}"

                # Add to displayed logs and update the log display area
                displayed_logs.append(formatted_log)
                log_display_area = displayed_logs[-log_display_height:] if len(displayed_logs) >= log_display_height else displayed_logs + [''] * (log_display_height - len(displayed_logs))

                # Store log for final display
                self.collected_logs.append((timestamp, log_msg))

            # Move cursor back up to the loading bar position (log_display_height + 1 lines up)
            print(f"\033[{log_display_height + 1}A", end='\r')

            # Print the loading bar
            print(f"\r{bar}\033[K")

            # Print the log display area with a semi-transparent overlay effect
            for log_line in log_display_area:
                # Add a subtle background to make logs appear behind a semi-transparent overlay
                if log_line:
                    print(f"\r{Back.BLACK}{Fore.WHITE}{log_line}{Style.RESET_ALL}\033[K")
                else:
                    print("\r\033[K")

            # Sleep for a short time
            time.sleep(sleep_time)

        # Move cursor back up to the loading bar position
        print(f"\033[{log_display_height + 1}A", end='\r')

        # Print completion message
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        completion_msg = f"{Fore.GREEN}[{timestamp}] ✓ {message} completed!{Style.RESET_ALL}"
        print(f"\r{completion_msg}\033[K")

        # Add completion log
        self.collected_logs.append((timestamp, f"✓ {message} completed!"))

        # Clear the log display area
        for _ in range(log_display_height):
            print("\r\033[K")

    def print_menu_with_settings(self, menu_items, title=None):
        """Print a menu with settings side by side.

        Args:
            menu_items (list): List of tuples (key, description)
            title (str, optional): Title for the menu
        """
        if title:
            self.print_header(title)

        # Calculate the maximum length of menu items for proper spacing
        max_menu_length = max([len(f"{key}: {desc}") for key, desc in menu_items]) + 5

        # Get settings items to display
        settings_items = [
            ("Current Settings", ""),
            ("Symbol", self.settings['symbol']),
            ("Timeframe", self.settings['timeframe']),
            ("Limit", self.settings['limit']),
            ("Exchange", self.settings['exchange']),
            ("Model Type", self.settings['model_type']),
            ("Model Path", self.settings['model_path'] or "None"),
            ("Theme", self.settings['theme']),
            ("Save", self.settings['save']),
            ("Tune", self.settings['tune']),
            ("Use LLM", self.settings['use_llm']),
            ("LLM Model", self.settings['llm_model']),
            ("Use GPU", self.settings['use_gpu'])
        ]

        # Determine how many menu items to show
        menu_count = len(menu_items)
        settings_count = len(settings_items)

        # Print menu items and settings side by side
        for i in range(max(menu_count, settings_count)):
            menu_str = ""
            settings_str = ""

            # Add menu item if available
            if i < menu_count:
                key, desc = menu_items[i]

                # Check if this menu item corresponds to a function that has been completed
                check_mark = ""
                if desc == "Fetch Data" and self.completed_functions.get('fetch_data'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Analyze Data" and self.completed_functions.get('analyze_data'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Train Model" and self.completed_functions.get('train_model'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Make Predictions" and self.completed_functions.get('make_predictions'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Backtest Strategy" and self.completed_functions.get('backtest_strategy'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "LLM Analysis" and self.completed_functions.get('llm_analysis'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"

                menu_str = f"{Fore.GREEN}{key}{Style.RESET_ALL}: {desc}{check_mark}"

            # Add settings item if available
            if i < settings_count:
                setting_key, setting_val = settings_items[i]
                if i == 0:  # This is the header
                    settings_str = f"{Fore.YELLOW}{setting_key}{Style.RESET_ALL}"
                else:
                    settings_str = f"{Fore.YELLOW}{setting_key}:{Style.RESET_ALL} {setting_val}"

            # Print the line with proper spacing
            print(f"{menu_str}{' ' * (max_menu_length - len(menu_str.replace(Fore.GREEN, '').replace(Style.RESET_ALL, '').replace('✓', '')))}  {settings_str}")

        print()

    def main_menu(self):
        """Display the main menu."""
        menu_items = [
            ("1", "Fetch Data"),
            ("2", "Analyze Data"),
            ("3", "Train Model"),
            ("4", "Make Predictions"),
            ("5", "Backtest Strategy"),
            ("6", "Run All Steps"),
            ("7", "LLM Analysis"),
            ("8", "Settings"),
            ("c", "Clear Data"),
            ("h", "How to Use"),
            ("q", "Quit")
        ]

        self.print_menu_with_settings(menu_items, "3lacks Scanner - Main Menu")

        choice = self.get_input("Enter your choice: ")

        if choice == '1':
            self.fetch_data()
        elif choice == '2':
            self.analyze_data()
        elif choice == '3':
            self.train_model()
        elif choice == '4':
            self.make_predictions()
        elif choice == '5':
            self.backtest_strategy()
        elif choice == '6':
            self.run_all_steps()
        elif choice == '7':
            self.llm_analysis()
        elif choice == '8':
            self.current_menu = self.settings_menu
        elif choice.lower() == 'c':
            self.clear_data()
        elif choice.lower() == 'h':
            self.show_how_to_use()
        elif choice.lower() == 'q':
            self.show_exit_screen()
            self.running = False
        else:
            self.print_error("Invalid choice. Please try again.")
            time.sleep(1)

    def settings_menu(self):
        """Display the settings menu."""
        menu_items = [
            ("1", "Change Symbol"),
            ("2", "Change Timeframe"),
            ("3", "Change Data Limit"),
            ("4", "Change Exchange"),
            ("5", "Change Model Type"),
            ("6", "Select Model Path"),
            ("7", "Change Theme"),
            ("8", "Toggle LLM Analysis"),
            ("9", "Change LLM Model"),
            ("g", "Toggle GPU Acceleration"),
            ("s", "Toggle Save Results"),
            ("t", "Toggle Hyperparameter Tuning"),
            ("b", "Back to Main Menu")
        ]

        self.print_menu_with_settings(menu_items, "3lacks Scanner - Settings")

        choice = self.get_input("Enter your choice: ")

        if choice == '1':
            self.change_symbol()
        elif choice == '2':
            self.change_timeframe()
        elif choice == '3':
            self.change_limit()
        elif choice == '4':
            self.change_exchange()
        elif choice == '5':
            self.change_model_type()
        elif choice == '6':
            self.select_model_path()
        elif choice == '7':
            self.change_theme()
        elif choice == '8':
            self.settings['use_llm'] = not self.settings['use_llm']
            self.print_success(f"LLM Analysis {'enabled' if self.settings['use_llm'] else 'disabled'}.")
            time.sleep(1)
        elif choice == '9':
            self.change_llm_model()
        elif choice == 'g':
            self.settings['use_gpu'] = not self.settings['use_gpu']
            self.print_success(f"GPU Acceleration {'enabled' if self.settings['use_gpu'] else 'disabled'}.")
            time.sleep(1)
        elif choice == 's':
            self.settings['save'] = not self.settings['save']
            self.print_success(f"Save results {'enabled' if self.settings['save'] else 'disabled'}.")
            time.sleep(1)
        elif choice == 't':
            self.settings['tune'] = not self.settings['tune']
            self.print_success(f"Hyperparameter tuning {'enabled' if self.settings['tune'] else 'disabled'}.")
            time.sleep(1)
        elif choice.lower() == 'b':
            self.current_menu = self.main_menu
        else:
            self.print_error("Invalid choice. Please try again.")
            time.sleep(1)

    def change_symbol(self):
        """Change the symbol setting."""
        self.print_header("Change Symbol")

        # Group symbols by quote currency for better organization
        print("Available symbols:")

        # Create a display list for easier selection
        display_symbols = []

        # First, organize by quote currency
        for quote_format in self.quote_formats:
            quote = quote_format['quote']
            print(f"\n{quote} pairs:")

            # Filter symbols for this quote currency
            # For USD pairs, we need to be more specific to avoid showing USDT pairs
            if quote == 'USD':
                quote_symbols = [s for s in self.available_symbols if s.endswith(f'-{quote}')]
            else:
                quote_symbols = [s for s in self.available_symbols if s.endswith(f'/{quote}')]
            quote_symbols.sort()  # Sort alphabetically

            # Display symbols for this quote currency
            for symbol in quote_symbols:
                display_symbols.append(symbol)
                print(f"{len(display_symbols)}. {symbol}")

        print(f"\nCurrent symbol: {self.settings['symbol']}")

        choice = self.get_input("\nEnter symbol number or custom symbol (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(display_symbols):
                self.settings['symbol'] = display_symbols[index]
                self.print_success(f"Symbol changed to {self.settings['symbol']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            # Custom symbol
            self.settings['symbol'] = choice
            self.print_success(f"Symbol changed to {self.settings['symbol']}.")

        time.sleep(1)

    def change_timeframe(self):
        """Change the timeframe setting."""
        self.print_header("Change Timeframe")

        print("Available timeframes:")
        for i, timeframe in enumerate(self.available_timeframes, 1):
            print(f"{i}. {timeframe}")

        print(f"\nCurrent timeframe: {self.settings['timeframe']}")

        choice = self.get_input("\nEnter timeframe number or custom timeframe (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(self.available_timeframes):
                self.settings['timeframe'] = self.available_timeframes[index]
                self.print_success(f"Timeframe changed to {self.settings['timeframe']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            # Custom timeframe
            self.settings['timeframe'] = choice
            self.print_success(f"Timeframe changed to {self.settings['timeframe']}.")

        time.sleep(1)

    def change_limit(self):
        """Change the data limit setting."""
        self.print_header("Change Data Limit")

        print(f"Current limit: {self.settings['limit']}")

        choice = self.get_input("\nEnter new limit (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            limit = int(choice)
            if limit > 0:
                self.settings['limit'] = limit
                self.print_success(f"Limit changed to {self.settings['limit']}.")
            else:
                self.print_error("Limit must be greater than 0.")
        except ValueError:
            self.print_error("Invalid input. Please enter a number.")

        time.sleep(1)

    def change_exchange(self):
        """Change the exchange setting."""
        self.print_header("Change Exchange")

        print("Available exchanges:")
        exchanges = ['coinbase', 'kraken', 'kucoin', 'huobi', 'bybit']
        for i, exchange in enumerate(exchanges, 1):
            print(f"{i}. {exchange}")

        print(f"\nCurrent exchange: {self.settings['exchange']}")

        choice = self.get_input("\nEnter exchange number or custom exchange (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(exchanges):
                self.settings['exchange'] = exchanges[index]
                self.print_success(f"Exchange changed to {self.settings['exchange']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            # Custom exchange
            self.settings['exchange'] = choice
            self.print_success(f"Exchange changed to {self.settings['exchange']}.")

        time.sleep(1)

    def change_model_type(self):
        """Change the model type setting."""
        self.print_header("Change Model Type")

        print("Available model types:")
        for i, model_type in enumerate(self.available_model_types, 1):
            print(f"{i}. {model_type}")

        print(f"\nCurrent model type: {self.settings['model_type']}")

        choice = self.get_input("\nEnter model type number (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(self.available_model_types):
                self.settings['model_type'] = self.available_model_types[index]
                self.print_success(f"Model type changed to {self.settings['model_type']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            self.print_error("Invalid input. Please enter a number.")

        time.sleep(1)

    def select_model_path(self):
        """Select a model path."""
        self.print_header("Select Model Path")

        # Refresh available models
        self.refresh_available_models()

        if not self.available_models:
            self.print_warning("No models found. Train a model first.")
            time.sleep(2)
            return

        print("Available models:")
        for i, model_path in enumerate(self.available_models, 1):
            print(f"{i}. {os.path.basename(model_path)}")

        print(f"\nCurrent model path: {self.settings['model_path']}")

        choice = self.get_input("\nEnter model number (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(self.available_models):
                self.settings['model_path'] = self.available_models[index]
                self.print_success(f"Model path changed to {self.settings['model_path']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            self.print_error("Invalid input. Please enter a number.")

        time.sleep(1)

    def change_theme(self):
        """Change the theme setting."""
        self.print_header("Change Theme")

        print("Available themes:")
        for i, theme in enumerate(self.available_themes, 1):
            print(f"{i}. {theme}")

        print(f"\nCurrent theme: {self.settings['theme']}")

        choice = self.get_input("\nEnter theme number (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(self.available_themes):
                self.settings['theme'] = self.available_themes[index]
                self.print_success(f"Theme changed to {self.settings['theme']}.")
            else:
                self.print_error("Invalid choice. Please try again.")
        except ValueError:
            self.print_error("Invalid input. Please enter a number.")

        time.sleep(1)

    def display_model_details(self, model_key, model_info):
        """Display detailed information about a model."""
        # Create a box for the model details
        width = 80
        print("\n" + "=" * width)
        print(f"{Fore.CYAN}{model_key.upper()} MODEL DETAILS{Style.RESET_ALL}".center(width))
        print("=" * width)

        # Basic information
        print(f"{Fore.YELLOW}Description:{Style.RESET_ALL} {model_info['description']}")
        print(f"{Fore.YELLOW}File Size:{Style.RESET_ALL} {model_info['size_gb']:.2f} GB")
        print(f"{Fore.YELLOW}File Name:{Style.RESET_ALL} {model_info['name']}")

        # Detailed information
        print("\n" + "-" * width)
        print(f"{Fore.CYAN}DETAILED INFORMATION{Style.RESET_ALL}".center(width))
        print("-" * width)
        print(f"{Fore.YELLOW}Overview:{Style.RESET_ALL} {model_info['details']}")

        # Trading focus
        trading_focus = model_info.get('trading_focus', 'Unknown')
        focus_color = Fore.GREEN if trading_focus in ['High', 'Very High'] else Fore.YELLOW if trading_focus == 'Medium' else Fore.RED
        print(f"{Fore.YELLOW}Trading Focus:{Style.RESET_ALL} {focus_color}{trading_focus}{Style.RESET_ALL}")

        # Hardware requirements
        print(f"{Fore.YELLOW}Hardware Requirements:{Style.RESET_ALL} {model_info.get('hardware_req', 'Not specified')}")

        # Strengths and weaknesses
        print("\n" + "-" * width)
        print(f"{Fore.CYAN}STRENGTHS & WEAKNESSES{Style.RESET_ALL}".center(width))
        print("-" * width)
        print(f"{Fore.GREEN}Strengths:{Style.RESET_ALL} {model_info.get('strengths', 'Not specified')}")
        print(f"{Fore.RED}Weaknesses:{Style.RESET_ALL} {model_info.get('weaknesses', 'Not specified')}")

        print("=" * width)

    def show_spinner_animation(self, message, callback_func, *args, **kwargs):
        """Show a spinner animation while executing a callback function.

        Args:
            message (str): Message to display with the spinner
            callback_func (callable): Function to execute while showing the spinner
            *args: Arguments to pass to the callback function
            **kwargs: Keyword arguments to pass to the callback function

        Returns:
            Any: The return value of the callback function
        """
        # Spinner characters
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        # Colors for the spinner
        colors = [Fore.CYAN, Fore.BLUE, Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.MAGENTA]

        # Clear the screen and print header
        self.clear_screen()
        self.print_header("3lacks Scanner - LLM Download")

        # Print initial message
        print(f"\n{Fore.CYAN}{message}{Style.RESET_ALL}\n")

        # Start the spinner in a separate thread
        import threading
        import time

        # Flag to control the spinner
        running = True
        result = None
        error = None

        # Function to run the callback in a separate thread
        def run_callback():
            nonlocal result, error, running
            try:
                result = callback_func(*args, **kwargs)
            except Exception as e:
                error = e
            finally:
                running = False

        # Start the callback thread
        callback_thread = threading.Thread(target=run_callback)
        callback_thread.daemon = True
        callback_thread.start()

        # Show the spinner while the callback is running
        i = 0
        progress_chars = [' '] * 20  # For the progress trail effect
        progress_pos = 0

        try:
            while running:
                # Get the current spinner character and color
                spinner = spinner_chars[i % len(spinner_chars)]
                color = colors[i % len(colors)]

                # Update progress trail
                progress_chars[progress_pos] = '·'
                progress_pos = (progress_pos + 1) % len(progress_chars)
                progress_chars[progress_pos] = '○'
                progress_trail = ''.join(progress_chars)

                # Print the spinner and message
                print(f"\r{color}{spinner}{Style.RESET_ALL} {message} {Fore.CYAN}{progress_trail}{Style.RESET_ALL}", end='')

                # Sleep briefly
                time.sleep(0.1)
                i += 1

                # Every 10 iterations, update the progress trail
                if i % 10 == 0:
                    for j in range(len(progress_chars)):
                        if progress_chars[j] == '·':
                            progress_chars[j] = ' '
        finally:
            # Clear the spinner line
            print("\r" + " " * 100, end="\r")

            # If there was an error, raise it
            if error:
                raise error

            # Return the result
            return result

    def check_llm_dependencies(self):
        """Check if all required LLM dependencies are installed and offer to install them if missing.

        Returns:
            bool: True if all dependencies are installed or successfully installed, False otherwise
        """
        try:
            from src.analysis.local_llm import check_dependencies, install_dependencies, REQUIRED_PACKAGES

            all_installed, missing_packages = check_dependencies()
            if all_installed:
                return True

            # Show missing dependencies
            self.print_header("Missing Dependencies")
            self.print_warning(f"The following required packages are missing: {', '.join(missing_packages)}")
            self.print_info("These packages are required for LLM functionality.")

            # Ask to install
            install = self.get_input("\nWould you like to install these packages now? (y/n): ")
            if install.lower() != 'y':
                self.print_info("Dependencies not installed. LLM functionality will be limited.")
                return False

            # Show installation progress
            self.print_info("Installing dependencies...")

            # Show loading animation for installation
            install_logs = [
                "Checking package versions...",
                "Resolving dependencies...",
                "Downloading packages...",
                "Installing packages...",
                "Building wheels...",
                "Verifying installations...",
                "Cleaning up...",
                "Finalizing installation..."
            ]
            self.show_loading_animation("Installing required packages", duration=3, log_messages=install_logs)

            # Actually install the dependencies
            success = install_dependencies(missing_packages)

            if success:
                self.print_success("Successfully installed all required dependencies!")
                return True
            else:
                self.print_error("Failed to install dependencies. Please install them manually:")
                self.print_info(f"pip install {' '.join(missing_packages)}")
                return False

        except Exception as e:
            self.print_error(f"Error checking dependencies: {str(e)}")
            return False

    def download_llm_model(self, model_key):
        """Download an LLM model.

        Args:
            model_key (str): The key of the model to download

        Returns:
            bool: True if download was successful, False otherwise
        """
        # First check dependencies
        if not self.check_llm_dependencies():
            self.print_warning("Cannot download model without required dependencies.")
            return False

        try:
            from src.analysis.local_llm import AVAILABLE_MODELS, DEFAULT_MODEL_PATH
            import os
            import requests
            import threading
            import time
            from tqdm import tqdm
            from huggingface_hub import hf_hub_download

            # Get model info
            if model_key not in AVAILABLE_MODELS:
                self.print_error(f"Unknown model: {model_key}")
                return False

            model_info = AVAILABLE_MODELS[model_key]
            model_name = model_info.get('name')
            model_url = model_info.get('url')
            model_size = model_info.get('size_gb', 0)

            # Create model directory if it doesn't exist
            model_dir = DEFAULT_MODEL_PATH
            os.makedirs(model_dir, exist_ok=True)

            # Full path to save the model
            model_file_path = os.path.join(model_dir, model_name)

            # Check if model already exists
            if os.path.exists(model_file_path):
                self.print_info(f"Model already exists at {model_file_path}")
                return True

            # Show download information
            self.print_header(f"Downloading {model_key}")
            self.print_info(f"Model: {model_name}")
            self.print_info(f"Size: {model_size:.2f} GB")
            self.print_info(f"Destination: {model_file_path}")
            self.print_info("This may take a while depending on your internet connection...")

            # Ask for confirmation
            confirm = self.get_input(f"\nStart download? (y/n): ")
            if confirm.lower() != 'y':
                self.print_info("Download cancelled.")
                return False

            # Variables to track download progress
            download_complete = False
            download_success = False
            download_error = None
            download_progress = 0
            total_size = 0

            # Function to perform the actual download
            def perform_download():
                nonlocal download_complete, download_success, download_error, download_progress, total_size
                try:
                    if model_url:
                        # Direct download with progress tracking
                        response = requests.get(model_url, stream=True)
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0

                        with open(model_file_path, 'wb') as f:
                            for data in response.iter_content(chunk_size=1024*1024):
                                if data:
                                    f.write(data)
                                    downloaded_size += len(data)
                                    download_progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    else:
                        # Download from HuggingFace
                        # Try to parse model name to get repo_id and filename
                        if "/" in model_name:
                            repo_id, filename = model_name.split("/", 1)
                        else:
                            # Default to TheBloke's repository
                            repo_id = "TheBloke/Llama-3-8B-Instruct-GGUF"
                            filename = model_name

                        # Download the model
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False
                        )

                    download_success = True
                except Exception as e:
                    download_error = str(e)
                    # Try to remove partially downloaded file
                    if os.path.exists(model_file_path):
                        try:
                            os.remove(model_file_path)
                        except:
                            pass
                finally:
                    download_complete = True

            # Start the download in a separate thread
            download_thread = threading.Thread(target=perform_download)
            download_thread.daemon = True
            download_thread.start()

            # Show spinner animation while downloading
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            colors = [Fore.CYAN, Fore.BLUE, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA]
            i = 0

            # Clear screen and show download header
            self.clear_screen()
            self.print_header(f"Downloading {model_key}")
            print(f"\n{Fore.CYAN}Model:{Style.RESET_ALL} {model_name}")
            print(f"{Fore.CYAN}Size:{Style.RESET_ALL} {model_size:.2f} GB")
            print(f"{Fore.CYAN}Destination:{Style.RESET_ALL} {model_file_path}\n")

            # Show spinner while downloading
            while not download_complete:
                spinner = spinner_chars[i % len(spinner_chars)]
                color = colors[i % len(colors)]

                # Calculate progress bar
                if model_url and total_size > 0:
                    bar_width = 40
                    filled_width = int(download_progress / 100 * bar_width)
                    bar = f"[{Fore.GREEN}{'█' * filled_width}{Style.RESET_ALL}{' ' * (bar_width - filled_width)}] {download_progress:.1f}%"
                    status = f"{color}{spinner}{Style.RESET_ALL} Downloading LLM model... {bar}"
                else:
                    status = f"{color}{spinner}{Style.RESET_ALL} Downloading LLM model..."

                print(f"\r{status}", end='')
                time.sleep(0.1)
                i += 1

            # Clear the spinner line
            print("\r" + " " * 100)

            # Check download result
            if download_success:
                self.print_success(f"Model downloaded successfully to {model_file_path}")
                return True
            else:
                self.print_error(f"Error downloading model: {download_error}")
                return False

        except ImportError as e:
            self.print_error(f"Missing dependencies: {str(e)}")
            self.print_info("Please install required packages: pip install huggingface-hub tqdm requests")
            return False
        except Exception as e:
            self.print_error(f"Error: {str(e)}")
            return False

    def change_llm_model(self):
        """Change the LLM model setting."""
        self.print_header("Change LLM Model")

        # Import available models from local_llm module
        try:
            from src.analysis.local_llm import AVAILABLE_MODELS

            # Available LLM models
            available_models = list(AVAILABLE_MODELS.keys())

            # Group models by trading focus
            trading_focused = []
            general_purpose = []
            for model_key in available_models:
                model_info = AVAILABLE_MODELS[model_key]
                trading_focus = model_info.get('trading_focus', 'Low')
                if trading_focus in ['High', 'Very High']:
                    trading_focused.append(model_key)
                else:
                    general_purpose.append(model_key)

            # Display trading-focused models first
            print(f"{Fore.CYAN}Trading-Focused Models:{Style.RESET_ALL}")
            for i, model_key in enumerate(trading_focused, 1):
                model_info = AVAILABLE_MODELS[model_key]
                focus = model_info.get('trading_focus', '')
                focus_color = Fore.GREEN if focus == 'Very High' else Fore.YELLOW
                print(f"{i}. {model_key} - {model_info['description']} ({model_info['size_gb']:.2f} GB) {focus_color}[{focus} Trading Focus]{Style.RESET_ALL}")

            # Display general-purpose models
            print(f"\n{Fore.CYAN}General-Purpose Models:{Style.RESET_ALL}")
            for i, model_key in enumerate(general_purpose, len(trading_focused) + 1):
                model_info = AVAILABLE_MODELS[model_key]
                print(f"{i}. {model_key} - {model_info['description']} ({model_info['size_gb']:.2f} GB)")

            # Combine lists for selection
            all_models = trading_focused + general_purpose

            print(f"\nCurrent LLM model: {Fore.YELLOW}{self.settings['llm_model']}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Options:{Style.RESET_ALL}")
            print(f"- Enter a model number to view details and select a model")
            print(f"- Enter '{Fore.GREEN}d{Style.RESET_ALL}' to download a model")
            print(f"- Enter '{Fore.GREEN}b{Style.RESET_ALL}' to go back")

            choice = self.get_input("\nEnter your choice: ")

            if choice.lower() == 'b':
                return
            elif choice.lower() == 'd':
                # Download a model
                download_choice = self.get_input("Enter model number to download: ")
                try:
                    index = int(download_choice) - 1
                    if 0 <= index < len(all_models):
                        selected_model = all_models[index]
                        self.download_llm_model(selected_model)
                        self.wait_for_key()
                    else:
                        self.print_error("Invalid choice. Please try again.")
                        time.sleep(1)
                except ValueError:
                    self.print_error("Invalid input. Please enter a number.")
                    time.sleep(1)
                return

            try:
                index = int(choice) - 1
                if 0 <= index < len(all_models):
                    selected_model = all_models[index]
                    model_info = AVAILABLE_MODELS[selected_model]

                    # Display detailed information about the selected model
                    self.display_model_details(selected_model, model_info)

                    # Ask for confirmation
                    confirm = self.get_input(f"\nDo you want to use the {Fore.YELLOW}{selected_model}{Style.RESET_ALL} model? (y/n): ")

                    if confirm.lower() == 'y':
                        self.settings['llm_model'] = selected_model
                        self.print_success(f"LLM model changed to {self.settings['llm_model']}.")

                        # Show download information
                        self.print_info(f"Note: This model will be downloaded ({model_info['size_gb']:.2f} GB) when first used.")

                        # Ask if user wants to download the model now
                        download_now = self.get_input("Would you like to download this model now? (y/n): ")
                        if download_now.lower() == 'y':
                            self.download_llm_model(selected_model)

                        # Show hardware requirements
                        hardware_req = model_info.get('hardware_req', 'Not specified')
                        self.print_info(f"Hardware Requirements: {hardware_req}")

                        # Recommend GPU if appropriate
                        if not self.settings['use_gpu'] and 'GPU' in hardware_req:
                            self.print_warning("This model would benefit from GPU acceleration. Consider enabling it in settings.")
                    else:
                        self.print_info("Model selection cancelled.")
                else:
                    self.print_error("Invalid choice. Please try again.")
            except ValueError:
                self.print_error("Invalid input. Please enter a number.")
        except ImportError:
            self.print_error("Could not import available models. Please check your installation.")
        except Exception as e:
            self.print_error(f"Error: {str(e)}")

        time.sleep(2)

    def build_command_args(self):
        """Build command line arguments from settings."""
        args = argparse.Namespace()

        # Copy settings to args
        for key, value in self.settings.items():
            setattr(args, key, value)

        # Ensure LLM arguments are properly set
        if hasattr(args, 'use_llm') and args.use_llm:
            args.use_llm = True
            args.llm_model = self.settings['llm_model']
            args.use_gpu = self.settings['use_gpu']

        return args

    def fetch_data(self):
        """Fetch data based on current settings."""
        self.print_header("Fetching Data")

        try:
            from main import fetch_data

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2)

            df = fetch_data(args)

            if df is not None:
                self.print_success(f"Successfully fetched {len(df)} rows of data.")
                self.print_info(f"Time range: {df.index.min()} to {df.index.max()}")
                self.print_info(f"Latest price: {df['close'].iloc[-1]:.2f}")
                # Mark this function as completed
                self.completed_functions['fetch_data'] = True
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error fetching data: {e}")

        self.wait_for_key()

    def analyze_data(self):
        """Analyze data based on current settings."""
        self.print_header("Analyzing Data")

        try:
            from main import fetch_data, analyze_data

            args = self.build_command_args()

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2)

            df = fetch_data(args)

            if df is not None:
                # Mark fetch_data as completed
                self.completed_functions['fetch_data'] = True

                self.print_info("Performing technical analysis...")

                # Show loading animation while analyzing data
                self.show_loading_animation("Calculating technical indicators", duration=3)

                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_success(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators.")
                    # Mark analyze_data as completed
                    self.completed_functions['analyze_data'] = True

                    # Show key indicators instead of sample data
                    latest = df_analyzed.iloc[-1]
                    print("\nKey indicators (latest values):")

                    # Show RSI if available
                    if 'rsi_14' in df_analyzed.columns:
                        rsi = latest['rsi_14']
                        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        print(f"RSI (14): {rsi:.2f} - {rsi_status}")

                    # Show MACD if available
                    if 'MACD_12_26_9' in df_analyzed.columns and 'MACDs_12_26_9' in df_analyzed.columns:
                        macd = latest['MACD_12_26_9']
                        signal = latest['MACDs_12_26_9']
                        macd_status = "Bullish" if macd > signal else "Bearish"
                        print(f"MACD: {macd:.4f}, Signal: {signal:.4f} - {macd_status}")

                    # Show Bollinger Bands if available
                    if 'BBL_20_2.0' in df_analyzed.columns and 'BBM_20_2.0' in df_analyzed.columns and 'BBU_20_2.0' in df_analyzed.columns:
                        lower = latest['BBL_20_2.0']
                        middle = latest['BBM_20_2.0']
                        upper = latest['BBU_20_2.0']
                        price = latest['close']
                        bb_status = "Below lower band" if price < lower else "Above upper band" if price > upper else "Within bands"
                        print(f"Bollinger Bands: Lower: {lower:.2f}, Middle: {middle:.2f}, Upper: {upper:.2f} - {bb_status}")

                    # Display analysis in current terminal
                    self.print_info("\nDisplaying detailed analysis...")

                    # Show loading animation while preparing output
                    self.show_loading_animation("Preparing analysis report", duration=1.5)

                    # Display the collected logs for this operation only
                    self.display_collected_logs(f"Analysis Process Log - {args.symbol}", operation_start_time)

                    # Display the analysis in the current terminal
                    if self.output_generator.display_in_current_terminal(
                        df_analyzed, args.symbol, args.timeframe, report_type='analysis'
                    ):
                        self.print_success("Analysis completed successfully.")
                    else:
                        self.print_error("Failed to display analysis output.")
                else:
                    self.print_error("Analysis resulted in empty DataFrame or failed.")
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error analyzing data: {e}")

        self.wait_for_key()

    def train_model(self):
        """Train a model based on current settings."""
        self.print_header("Training Model")

        try:
            from main import fetch_data, analyze_data, train_model

            args = self.build_command_args()

            # Clear any previous logs
            self.collected_logs = []

            # Fetch data with minimal output
            self.print_info(f"Fetching data for {args.symbol}...")

            # Define custom logs for data fetching
            fetch_logs = [
                "Connecting to exchange API...",
                "Downloading historical data...",
                "Processing candles...",
                "Validating data integrity...",
                "Preparing dataset..."
            ]
            self.show_loading_animation("Retrieving market data", duration=2, log_messages=fetch_logs)

            df = fetch_data(args)

            if df is not None:
                # Analyze data with minimal output
                self.print_info("Calculating technical indicators...")

                # Define custom logs for analysis
                analysis_logs = [
                    "Computing price indicators...",
                    "Calculating momentum oscillators...",
                    "Generating trend metrics...",
                    "Evaluating volatility measures...",
                    "Creating trading signals..."
                ]
                self.show_loading_animation("Analyzing market data", duration=2.5, log_messages=analysis_logs)

                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    # Train model with enhanced loading animation
                    self.print_info(f"Training {args.model_type.replace('_', ' ').title()} model...")

                    # Define custom logs for model training (reduced verbosity)
                    train_logs = [
                        "Preparing dataset...",
                        "Initializing model...",
                        "Training model...",
                        "Evaluating performance..."
                    ]
                    self.show_loading_animation("Training machine learning model", duration=4, log_messages=train_logs)

                    model, model_path = train_model(df_analyzed, args)

                    if model is not None:
                        # Display the collected logs (concise version)
                        self.display_collected_logs(f"Model Training Summary - {args.symbol}")

                        self.print_success("\nModel trained successfully")
                        # Mark train_model as completed
                        self.completed_functions['train_model'] = True

                        # Show minimal model information
                        feature_count = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown'

                        # Create a compact summary box
                        print("\n" + "=" * 30)
                        print(f"{Fore.CYAN}MODEL SUMMARY{Style.RESET_ALL}")
                        print("=" * 30)
                        print(f"Type: {Fore.YELLOW}{args.model_type.replace('_', ' ').title()}{Style.RESET_ALL}")
                        print(f"Features: {Fore.YELLOW}{feature_count}{Style.RESET_ALL}")

                        if model_path:
                            print(f"Model: {Fore.GREEN}{os.path.basename(model_path)}{Style.RESET_ALL}")
                            self.settings['model_path'] = model_path
                            self.refresh_available_models()
                    else:
                        self.print_error("Failed to train model.")
                else:
                    self.print_error("Analysis resulted in empty DataFrame or failed.")
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error training model: {e}")

        self.wait_for_key()

    def make_predictions(self):
        """Make predictions based on current settings."""
        self.print_header("Making Predictions")

        if self.settings['model_path'] is None:
            self.print_error("No model path selected. Please select a model path in Settings.")
            self.wait_for_key()
            return

        try:
            from main import fetch_data, analyze_data, predict

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show enhanced loading animation with custom logs
            fetch_logs = [
                "Connecting to exchange API...",
                "Authenticating connection...",
                "Setting up data parameters...",
                "Requesting historical data...",
                "Downloading candles...",
                "Processing OHLCV data...",
                "Validating timestamps...",
                "Formatting data structures...",
                "Checking for missing values...",
                "Finalizing data retrieval..."
            ]
            self.show_loading_animation("Retrieving market data", duration=2, log_messages=fetch_logs)

            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")

                # Show loading animation while analyzing data
                self.show_loading_animation("Calculating technical indicators", duration=2.5)

                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info("Making predictions...")

                    # Show enhanced loading animation with custom logs
                    predict_logs = [
                        "Loading trained model...",
                        "Preparing feature data...",
                        "Normalizing inputs...",
                        "Running model inference...",
                        "Calculating prediction probabilities...",
                        "Generating trading signals...",
                        "Evaluating prediction confidence...",
                        "Applying risk filters...",
                        "Formatting prediction results...",
                        "Finalizing predictions..."
                    ]
                    self.show_loading_animation("Running prediction model", duration=2, log_messages=predict_logs)

                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_success("Successfully made predictions.")
                        # Mark make_predictions as completed
                        self.completed_functions['make_predictions'] = True

                        # Display prediction summary instead of sample data
                        print("\nPrediction Summary:")

                        if 'prediction' in df_predictions.columns:
                            # Count predictions
                            pred_counts = df_predictions['prediction'].value_counts()
                            total_preds = len(df_predictions)

                            # Calculate percentages
                            bullish_count = pred_counts.get(1, 0)
                            bearish_count = pred_counts.get(0, 0)
                            bullish_pct = (bullish_count / total_preds) * 100 if total_preds > 0 else 0
                            bearish_pct = (bearish_count / total_preds) * 100 if total_preds > 0 else 0

                            # Display summary
                            print(f"Bullish signals: {bullish_count} ({bullish_pct:.1f}%)")
                            print(f"Bearish signals: {bearish_count} ({bearish_pct:.1f}%)")

                            # Latest prediction
                            latest_pred = df_predictions['prediction'].iloc[-1]
                            latest_prob = df_predictions['prediction_probability'].iloc[-1] if 'prediction_probability' in df_predictions.columns else None

                            pred_text = "Bullish" if latest_pred == 1 else "Bearish"
                            prob_text = f" (Confidence: {latest_prob:.2f})" if latest_prob is not None else ""

                            print(f"\nLatest prediction: {Fore.YELLOW}{pred_text}{prob_text}{Style.RESET_ALL}")
                            print(f"Current price: {df_predictions['close'].iloc[-1]:.2f}")
                        else:
                            print("No prediction column found in results.")

                        # Display predictions in current terminal
                        self.print_info("\nDisplaying detailed predictions...")

                        # Show loading animation while preparing output
                        self.show_loading_animation("Preparing prediction report", duration=1.5)

                        # Display the collected logs for this operation only
                        self.display_collected_logs(f"Prediction Process Log - {args.symbol}", operation_start_time)

                        # Display the predictions in the current terminal
                        if self.output_generator.display_in_current_terminal(
                            df_predictions, args.symbol, args.timeframe, report_type='prediction'
                        ):
                            self.print_success("Predictions displayed successfully.")
                        else:
                            self.print_error("Failed to display predictions output.")
                    else:
                        self.print_error("Failed to make predictions.")
                else:
                    self.print_error("Analysis resulted in empty DataFrame or failed.")
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error making predictions: {e}")

        self.wait_for_key()

    def backtest_strategy(self):
        """Backtest a strategy based on current settings."""
        self.print_header("Backtesting Strategy")

        if self.settings['model_path'] is None:
            self.print_error("No model path selected. Please select a model path in Settings.")
            self.wait_for_key()
            return

        try:
            from main import fetch_data, analyze_data, predict, backtest

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2)

            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")

                # Show loading animation while analyzing data
                self.show_loading_animation("Calculating technical indicators", duration=2.5)

                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info("Making predictions...")

                    # Show loading animation while making predictions
                    self.show_loading_animation("Running prediction model", duration=2)

                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_info("Backtesting predictions...")

                        # Show loading animation while backtesting
                        self.show_loading_animation("Simulating trading strategy", duration=3)

                        performance_metrics, trading_metrics = backtest(df_predictions, args)

                        # Display backtest results in a more visual way
                        print("\nBacktest Results:")

                        if performance_metrics:
                            self.print_success("Performance Metrics:")
                            # Format key metrics with colors
                            for key, value in performance_metrics.items():
                                if key == 'accuracy' or key == 'f1_score':
                                    formatted_value = f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"
                                elif 'profit' in key.lower() or 'return' in key.lower():
                                    color = Fore.GREEN if float(value) > 0 else Fore.RED
                                    formatted_value = f"{color}{value}{Style.RESET_ALL}"
                                else:
                                    formatted_value = f"{value}"
                                print(f"  {key}: {formatted_value}")

                        if trading_metrics:
                            self.print_success("\nTrading Metrics:")
                            # Format key metrics with colors
                            for key, value in trading_metrics.items():
                                if key == 'win_rate' or key == 'profit_factor':
                                    color = Fore.GREEN if float(value) > 1.5 else Fore.YELLOW if float(value) > 1 else Fore.RED
                                    formatted_value = f"{color}{value}{Style.RESET_ALL}"
                                elif 'profit' in key.lower() or 'return' in key.lower():
                                    color = Fore.GREEN if float(value) > 0 else Fore.RED
                                    formatted_value = f"{color}{value}{Style.RESET_ALL}"
                                else:
                                    formatted_value = f"{value}"
                                print(f"  {key}: {formatted_value}")

                        # Display backtest results in current terminal
                        self.print_info("\nDisplaying detailed backtest results...")

                        # Show loading animation while preparing output
                        self.show_loading_animation("Preparing backtest report", duration=1.5)

                        # Display the collected logs for this operation only
                        self.display_collected_logs(f"Backtest Process Log - {args.symbol}", operation_start_time)

                        # Display the backtest results in the current terminal
                        if self.output_generator.display_in_current_terminal(
                            df_predictions, args.symbol, args.timeframe, report_type='backtest',
                            performance_metrics=performance_metrics, trading_metrics=trading_metrics
                        ):
                            self.print_success("Backtest results displayed successfully.")
                            # Mark backtest_strategy as completed
                            self.completed_functions['backtest_strategy'] = True
                        else:
                            self.print_error("Failed to display backtest output.")
                    else:
                        self.print_error("Failed to make predictions.")
                else:
                    self.print_error("Analysis resulted in empty DataFrame or failed.")
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error backtesting strategy: {e}")

        self.wait_for_key()

    def llm_analysis(self):
        """Perform LLM analysis on market data."""
        self.print_header("LLM Analysis")

        # First check dependencies
        if not self.check_llm_dependencies():
            self.print_warning("Cannot perform LLM analysis without required dependencies.")
            self.print_info("Please install the required dependencies and try again.")
            self.wait_for_key()
            return

        try:
            from main import fetch_data, analyze_data
            from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS, DEFAULT_MODEL_PATH
            import os

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            args = self.build_command_args()

            # Check if LLM analysis is enabled
            if not self.settings['use_llm']:
                self.print_warning("LLM Analysis is currently disabled. Enable it in Settings > Toggle LLM Analysis.")
                self.wait_for_key()
                return

            # Fetch data
            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is None:
                self.print_error("Failed to fetch data.")
                self.wait_for_key()
                return

            # Analyze data
            self.print_info("Performing technical analysis...")
            df_analyzed = analyze_data(df, args)

            if df_analyzed is None or df_analyzed.empty:
                self.print_error("Analysis resulted in empty DataFrame or failed.")
                self.wait_for_key()
                return

            # Set DataFrame attributes for better context
            df_analyzed.attrs["symbol"] = args.symbol
            df_analyzed.attrs["timeframe"] = args.timeframe

            # Get model info
            model_info = AVAILABLE_MODELS.get(self.settings['llm_model'], {})
            model_name = model_info.get('name', self.settings['llm_model'])

            # Check if model exists, offer to download if not
            model_file_path = os.path.join(DEFAULT_MODEL_PATH, model_name)
            if not os.path.exists(model_file_path):
                self.print_warning(f"Model file not found: {model_name}")
                download = self.get_input(f"Would you like to download the {self.settings['llm_model']} model now? (y/n): ")
                if download.lower() == 'y':
                    success = self.download_llm_model(self.settings['llm_model'])
                    if not success:
                        self.print_error("Failed to download model. Cannot proceed with LLM analysis.")
                        self.wait_for_key()
                        return
                else:
                    self.print_info("Model download skipped. Cannot proceed with LLM analysis.")
                    self.wait_for_key()
                    return

            # Initialize LLM analyzer
            self.print_info(f"Initializing local LLM analyzer with model {self.settings['llm_model']}...")
            self.print_info(f"This may take a moment to load the model ({model_info.get('size_gb', 'unknown')} GB)")

            # Show enhanced loading animation for model initialization
            model_logs = [
                "Checking model availability...",
                "Downloading model files...",
                "Verifying model integrity...",
                "Loading model weights...",
                "Initializing neural network...",
                "Setting up inference engine...",
                "Optimizing memory usage...",
                "Configuring model parameters...",
                "Preparing model context...",
                "Model ready for inference..."
            ]
            self.show_loading_animation("Loading AI model", duration=3, log_messages=model_logs)

            # Configure GPU usage
            n_gpu_layers = 0
            if self.settings['use_gpu']:
                self.print_info("GPU acceleration enabled. Using GPU for inference.")
                n_gpu_layers = -1  # Use all layers on GPU

            # Initialize the analyzer
            llm_analyzer = LocalLLMAnalyzer(
                model_name=model_name,
                n_gpu_layers=n_gpu_layers
            )

            # Check if LLM was initialized successfully
            if llm_analyzer.llm is None:
                self.print_error("Failed to initialize LLM. Check the logs for details.")
                self.wait_for_key()
                return

            # Perform LLM analysis
            self.print_info("Analyzing market data with local LLM...")

            # Show enhanced loading animation for LLM processing
            analysis_logs = [
                "Preparing market data for analysis...",
                "Formatting prompt for LLM...",
                "Sending data to neural network...",
                "Processing market patterns...",
                "Analyzing price action...",
                "Evaluating technical indicators...",
                "Generating trading insights...",
                "Calculating risk assessment...",
                "Formulating trading recommendation...",
                "Finalizing AI analysis..."
            ]
            self.show_loading_animation("AI analyzing market data", duration=3, log_messages=analysis_logs)
            recommendation = llm_analyzer.analyze(df_analyzed)

            if "error" in recommendation:
                self.print_error(f"LLM analysis failed: {recommendation['error']}")
                self.wait_for_key()
                return

            # Save analysis if requested
            if self.settings['save']:
                filename = llm_analyzer.save_analysis(recommendation, args.symbol, args.timeframe)
                if filename:
                    self.print_success(f"Saved LLM analysis to {filename}")

            # Display the collected logs for this operation only
            self.display_collected_logs(f"LLM Analysis Process Log - {args.symbol}", operation_start_time)

            # Mark llm_analysis as completed
            self.completed_functions['llm_analysis'] = True

            # Display results
            self.print_header(f"LLM Analysis Results - {args.symbol} ({args.timeframe})")

            # Print recommendation summary
            self.print_info(f"\nTrading Recommendation: {Fore.YELLOW}{recommendation['recommendation']}{Style.RESET_ALL}")
            self.print_info(f"Risk Assessment: {Fore.YELLOW}{recommendation['risk']}{Style.RESET_ALL}")
            self.print_info(f"Model Used: {Fore.CYAN}{recommendation.get('model', model_name)}{Style.RESET_ALL}")
            self.print_info(f"Analysis Timestamp: {recommendation['timestamp']}")

            # Print entry/exit levels if available
            print("\n" + "=" * 40)
            print(f"{Fore.CYAN}ENTRY/EXIT LEVELS{Style.RESET_ALL}")
            print("=" * 40)

            # Entry price
            entry_price = recommendation.get('entry_price')
            if entry_price is not None:
                print(f"{Fore.GREEN}Entry Price:{Style.RESET_ALL} {entry_price:.2f}")
            else:
                print(f"{Fore.GREEN}Entry Price:{Style.RESET_ALL} Not specified")

            # Stop loss
            stop_loss = recommendation.get('stop_loss')
            if stop_loss is not None:
                print(f"{Fore.RED}Stop Loss:{Style.RESET_ALL} {stop_loss:.2f}")
            else:
                print(f"{Fore.RED}Stop Loss:{Style.RESET_ALL} Not specified")

            # Take profit
            take_profit = recommendation.get('take_profit')
            if take_profit is not None:
                print(f"{Fore.GREEN}Take Profit:{Style.RESET_ALL} {take_profit:.2f}")
            else:
                print(f"{Fore.GREEN}Take Profit:{Style.RESET_ALL} Not specified")

            # Risk/reward ratio
            risk_reward = recommendation.get('risk_reward')
            if risk_reward is not None:
                print(f"{Fore.YELLOW}Risk/Reward Ratio:{Style.RESET_ALL} {risk_reward}")
            else:
                print(f"{Fore.YELLOW}Risk/Reward Ratio:{Style.RESET_ALL} Not specified")

            print()

            # Print full analysis
            print(recommendation['analysis'])

            self.wait_for_key()

        except Exception as e:
            self.print_error(f"Error performing LLM analysis: {e}")
            import traceback
            self.print_error(traceback.format_exc())
            self.wait_for_key()

    def run_all_steps(self):
        """Run all steps based on current settings."""
        self.print_header("Running All Steps")

        try:
            from main import fetch_data, analyze_data, train_model, predict, backtest

            args = self.build_command_args()

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            # Fetch data
            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2)

            df = fetch_data(args)
            if df is None:
                self.print_error("Failed to fetch data.")
                self.wait_for_key()
                return

            self.print_success(f"Successfully fetched {len(df)} rows of data.")
            self.print_info(f"Time range: {df.index.min()} to {df.index.max()}")

            # Analyze data
            self.print_info("Performing technical analysis...")

            # Show enhanced loading animation with custom logs
            custom_logs = [
                "Initializing technical analysis...",
                "Loading price data...",
                "Calculating moving averages...",
                "Computing RSI indicators...",
                "Generating MACD signals...",
                "Calculating Bollinger Bands...",
                "Identifying support/resistance levels...",
                "Detecting chart patterns...",
                "Evaluating trend strength...",
                "Finalizing technical indicators..."
            ]
            self.show_loading_animation("Calculating technical indicators", duration=2.5, log_messages=custom_logs)

            df_analyzed = analyze_data(df, args)
            if df_analyzed is None or df_analyzed.empty:
                self.print_error("Analysis resulted in empty DataFrame or failed.")
                self.wait_for_key()
                return

            self.print_success(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators.")

            # Train model with enhanced loading animation
            self.print_info(f"Training {args.model_type.replace('_', ' ').title()} model...")

            # Define custom logs for model training
            train_logs = [
                "Preparing training dataset...",
                "Splitting data into train/test sets...",
                "Scaling features...",
                "Initializing model architecture...",
                "Training model on historical data...",
                "Optimizing model parameters...",
                "Evaluating model performance...",
                "Finalizing model..."
            ]
            self.show_loading_animation("Training machine learning model", duration=4, log_messages=train_logs)

            model, model_path = train_model(df_analyzed, args)
            if model is None:
                self.print_error("Failed to train model.")
                self.wait_for_key()
                return

            self.print_success("Model trained successfully")

            # Show concise model information
            feature_count = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown'
            print(f"Type: {args.model_type.replace('_', ' ').title()} | Features: {feature_count} | Tuning: {'Enabled' if args.tune else 'Disabled'}")

            if model_path:
                self.print_info(f"Model saved as: {os.path.basename(model_path)}")
                self.settings['model_path'] = model_path
                self.refresh_available_models()

            # Make predictions
            self.print_info("Making predictions...")

            # Show loading animation while making predictions
            self.show_loading_animation("Running prediction model", duration=2)

            df_predictions = predict(df_analyzed, model_path, args)
            if df_predictions is None:
                self.print_error("Failed to make predictions.")
                self.wait_for_key()
                return

            self.print_success("Successfully made predictions.")

            # Display prediction summary instead of sample data
            print("\nPrediction Summary:")

            if 'prediction' in df_predictions.columns:
                # Count predictions
                pred_counts = df_predictions['prediction'].value_counts()
                total_preds = len(df_predictions)

                # Calculate percentages
                bullish_count = pred_counts.get(1, 0)
                bearish_count = pred_counts.get(0, 0)
                bullish_pct = (bullish_count / total_preds) * 100 if total_preds > 0 else 0
                bearish_pct = (bearish_count / total_preds) * 100 if total_preds > 0 else 0

                # Display summary
                print(f"Bullish signals: {bullish_count} ({bullish_pct:.1f}%)")
                print(f"Bearish signals: {bearish_count} ({bearish_pct:.1f}%)")

            # Backtest
            self.print_info("Backtesting predictions...")

            # Show loading animation while backtesting
            self.show_loading_animation("Simulating trading strategy", duration=3)

            performance_metrics, trading_metrics = backtest(df_predictions, args)

            # Display backtest results in a more visual way
            print("\nBacktest Results:")

            if performance_metrics:
                self.print_success("Performance Metrics:")
                # Format key metrics with colors
                for key, value in performance_metrics.items():
                    if key == 'accuracy' or key == 'f1_score':
                        formatted_value = f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"
                    elif 'profit' in key.lower() or 'return' in key.lower():
                        color = Fore.GREEN if float(value) > 0 else Fore.RED
                        formatted_value = f"{color}{value}{Style.RESET_ALL}"
                    else:
                        formatted_value = f"{value}"
                    print(f"  {key}: {formatted_value}")

            if trading_metrics:
                self.print_success("\nTrading Metrics:")
                # Format key metrics with colors
                for key, value in trading_metrics.items():
                    if key == 'win_rate' or key == 'profit_factor':
                        color = Fore.GREEN if float(value) > 1.5 else Fore.YELLOW if float(value) > 1 else Fore.RED
                        formatted_value = f"{color}{value}{Style.RESET_ALL}"
                    elif 'profit' in key.lower() or 'return' in key.lower():
                        color = Fore.GREEN if float(value) > 0 else Fore.RED
                        formatted_value = f"{color}{value}{Style.RESET_ALL}"
                    else:
                        formatted_value = f"{value}"
                    print(f"  {key}: {formatted_value}")

            # Display complete analysis in current terminal
            self.print_info("\nDisplaying comprehensive analysis...")

            # Show loading animation while preparing output
            self.show_loading_animation("Preparing comprehensive report", duration=2)

            # Display the complete analysis in the current terminal
            if self.output_generator.display_in_current_terminal(
                df_predictions, args.symbol, args.timeframe, report_type='all',
                performance_metrics=performance_metrics, trading_metrics=trading_metrics
            ):
                self.print_success("Complete analysis displayed successfully.")
            else:
                self.print_error("Failed to display analysis output.")

            # Perform LLM analysis if enabled
            if args.use_llm:
                self.print_info("\nPerforming LLM analysis...")

                # Show loading animation for LLM analysis
                self.show_loading_animation("Initializing AI analysis", duration=2)

                from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS

                # Get model info
                model_info = AVAILABLE_MODELS.get(args.llm_model, {})
                model_name = model_info.get('name', args.llm_model)

                # Configure GPU usage
                n_gpu_layers = 0
                if args.use_gpu:
                    self.print_info("GPU acceleration enabled for LLM inference.")
                    n_gpu_layers = -1  # Use all layers on GPU

                # Initialize the analyzer
                llm_analyzer = LocalLLMAnalyzer(
                    model_name=model_name,
                    n_gpu_layers=n_gpu_layers
                )

                # Show loading animation for LLM processing
                self.show_loading_animation("AI analyzing market data", duration=3)

                recommendation = llm_analyzer.analyze(df_analyzed)

                if recommendation and "error" not in recommendation:
                    self.print_success("\nLLM Analysis Results:")
                    print(f"Trading Recommendation: {Fore.YELLOW}{recommendation['recommendation']}{Style.RESET_ALL}")
                    print(f"Risk Assessment: {Fore.YELLOW}{recommendation['risk']}{Style.RESET_ALL}")

                    # Print entry/exit levels if available
                    entry_price = recommendation.get('entry_price')
                    stop_loss = recommendation.get('stop_loss')
                    take_profit = recommendation.get('take_profit')
                    risk_reward = recommendation.get('risk_reward')

                    if any(x is not None for x in [entry_price, stop_loss, take_profit, risk_reward]):
                        print("\n" + "=" * 40)
                        print(f"{Fore.CYAN}ENTRY/EXIT LEVELS{Style.RESET_ALL}")
                        print("=" * 40)

                        if entry_price is not None:
                            print(f"{Fore.GREEN}Entry Price:{Style.RESET_ALL} {entry_price:.2f}")

                        if stop_loss is not None:
                            print(f"{Fore.RED}Stop Loss:{Style.RESET_ALL} {stop_loss:.2f}")

                        if take_profit is not None:
                            print(f"{Fore.GREEN}Take Profit:{Style.RESET_ALL} {take_profit:.2f}")

                        if risk_reward is not None:
                            print(f"{Fore.YELLOW}Risk/Reward Ratio:{Style.RESET_ALL} {risk_reward}")
                else:
                    error_msg = recommendation.get("error", "Unknown error") if recommendation else "Failed to get recommendation"
                    self.print_warning(f"LLM analysis skipped: {error_msg}")

            # Mark all functions as completed
            self.completed_functions['fetch_data'] = True
            self.completed_functions['analyze_data'] = True
            self.completed_functions['train_model'] = True
            self.completed_functions['make_predictions'] = True
            self.completed_functions['backtest_strategy'] = True
            if args.use_llm:
                self.completed_functions['llm_analysis'] = True

            self.print_success("\nSuccessfully completed all operations.")

        except Exception as e:
            self.print_error(f"Error running all steps: {e}")

        self.wait_for_key()

    def display_collected_logs(self, title="Process Logs", operation_start_time=None):
        """Display all collected logs in a stylish format.

        Args:
            title (str): Title for the log display
            operation_start_time (datetime, optional): If provided, only show logs after this time
        """
        if not hasattr(self, 'collected_logs') or not self.collected_logs:
            return

        # Print header
        width = 80
        print("\n" + "=" * width)
        print(f"{Fore.CYAN}{title:^{width}}{Style.RESET_ALL}")
        print("=" * width)

        # Filter logs by operation time if provided
        filtered_logs = self.collected_logs
        if operation_start_time:
            # Convert string timestamps to datetime objects for comparison
            filtered_logs = []
            for log in self.collected_logs:
                try:
                    # Parse timestamp (format: HH:MM:SS.mmm)
                    log_time_str = log[0]
                    # Get current date and combine with log time
                    today = datetime.now().strftime("%Y-%m-%d")
                    log_datetime = datetime.strptime(f"{today} {log_time_str}", "%Y-%m-%d %H:%M:%S.%f")

                    # Only include logs after operation start time
                    if log_datetime >= operation_start_time:
                        filtered_logs.append(log)
                except Exception:
                    # If parsing fails, include the log anyway
                    filtered_logs.append(log)

        # Remove duplicate logs (same timestamp and message)
        unique_logs = []
        seen = set()
        for log in filtered_logs:
            log_str = f"{log[0]}:{log[1]}"
            if log_str not in seen:
                seen.add(log_str)
                unique_logs.append(log)

        # Print logs with alternating background colors
        for i, (timestamp, message) in enumerate(unique_logs):
            # Choose color based on message content
            if "error" in message.lower() or "failed" in message.lower():
                color = Fore.RED
            elif "warning" in message.lower():
                color = Fore.YELLOW
            elif "completed" in message.lower() or "success" in message.lower() or "✓" in message:
                color = Fore.GREEN
            else:
                # Alternate between colors for regular messages
                colors = [Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
                color = colors[i % len(colors)]

            # Print formatted log entry
            print(f"{Fore.WHITE}[{timestamp}]{Style.RESET_ALL} {color}{message}{Style.RESET_ALL}")

        # Clear logs after displaying them
        self.collected_logs = []

        print("=" * width)
        print()

        # Clear logs after displaying
        self.collected_logs = []

    def show_how_to_use(self):
        """Display the 'How to Use' guide."""
        self.clear_screen()
        width = 80
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        title = "3lacks Scanner - How to Use"
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()

        # Check if the guide file exists
        guide_path = "docs/how_to_use.md"
        if not os.path.exists(guide_path):
            self.print_error(f"Guide file not found: {guide_path}")
            self.wait_for_key()
            return

        try:
            # Read the guide file
            with open(guide_path, 'r') as f:
                guide_content = f.readlines()

            # Display the guide with pagination
            lines_per_page = 20
            total_pages = (len(guide_content) + lines_per_page - 1) // lines_per_page
            current_page = 1

            while current_page <= total_pages:
                self.clear_screen()
                print(Fore.CYAN + "=" * width + Style.RESET_ALL)
                print(Fore.CYAN + f"{title} (Page {current_page}/{total_pages}){' ':^{width-len(title)-15}}" + Style.RESET_ALL)
                print(Fore.CYAN + "=" * width + Style.RESET_ALL)
                print()

                # Calculate start and end lines for current page
                start_line = (current_page - 1) * lines_per_page
                end_line = min(start_line + lines_per_page, len(guide_content))

                # Display current page content
                for line in guide_content[start_line:end_line]:
                    # Format markdown headings
                    if line.startswith('# '):
                        print(f"{Fore.YELLOW}{line.strip()}{Style.RESET_ALL}")
                    elif line.startswith('## '):
                        print(f"{Fore.GREEN}{line.strip()}{Style.RESET_ALL}")
                    elif line.startswith('### '):
                        print(f"{Fore.CYAN}{line.strip()}{Style.RESET_ALL}")
                    elif line.startswith('- '):
                        print(f"{Fore.BLUE}•{Style.RESET_ALL} {line[2:].strip()}")
                    elif line.startswith('**'):
                        print(f"{Fore.MAGENTA}{line.strip()}{Style.RESET_ALL}")
                    else:
                        print(line.rstrip())

                print()
                print(f"Page {current_page} of {total_pages}")
                print(f"[{Fore.GREEN}N{Style.RESET_ALL}]ext page, [{Fore.GREEN}P{Style.RESET_ALL}]revious page, [{Fore.GREEN}Q{Style.RESET_ALL}]uit")

                # Get user input for navigation
                nav = self.get_input("Enter choice: ").lower()
                if nav == 'n' and current_page < total_pages:
                    current_page += 1
                elif nav == 'p' and current_page > 1:
                    current_page -= 1
                elif nav == 'q':
                    break

        except Exception as e:
            self.print_error(f"Error displaying guide: {e}")

        self.wait_for_key()

    def clear_data(self):
        """Clear data files with confirmation."""
        self.print_header("Clear Data")

        print(f"{Fore.YELLOW}WARNING: This will permanently delete data files.{Style.RESET_ALL}")
        print("\nSelect what to clear:")
        print(f"{Fore.GREEN}1{Style.RESET_ALL}: Raw data files (data directory)")
        print(f"{Fore.GREEN}2{Style.RESET_ALL}: Analysis results (results directory)")
        print(f"{Fore.GREEN}3{Style.RESET_ALL}: Trained models (models directory)")
        print(f"{Fore.GREEN}4{Style.RESET_ALL}: All data (everything above)")
        print(f"{Fore.GREEN}b{Style.RESET_ALL}: Back to main menu")

        choice = self.get_input("\nEnter your choice: ")

        if choice.lower() == 'b':
            return

        # Define directories to clear based on user choice
        directories_to_clear = []
        if choice == '1':
            directories_to_clear = ['data']
            confirm_message = "Are you sure you want to delete all raw data files?"
        elif choice == '2':
            directories_to_clear = ['results']
            confirm_message = "Are you sure you want to delete all analysis results?"
        elif choice == '3':
            directories_to_clear = ['models']
            confirm_message = "Are you sure you want to delete all trained models?"
        elif choice == '4':
            directories_to_clear = ['data', 'results', 'models']
            confirm_message = "Are you sure you want to delete ALL data files, results, and models?"
        else:
            self.print_error("Invalid choice. Please try again.")
            time.sleep(1)
            return

        # Ask for confirmation
        print(f"\n{Fore.RED}{confirm_message}{Style.RESET_ALL}")
        confirm = self.get_input("Type 'yes' to confirm: ")

        if confirm.lower() != 'yes':
            self.print_info("Operation cancelled.")
            self.wait_for_key()
            return

        # Clear the selected directories
        files_deleted = 0
        for directory in directories_to_clear:
            if os.path.exists(directory):
                # Show loading animation
                self.print_info(f"\nClearing {directory} directory...")

                # Get list of files
                files = [os.path.join(directory, f) for f in os.listdir(directory)
                         if os.path.isfile(os.path.join(directory, f))]

                if not files:
                    self.print_info(f"No files found in {directory} directory.")
                    continue

                # Show loading animation while deleting files
                delete_logs = [
                    f"Scanning {directory} directory...",
                    f"Found {len(files)} files to delete...",
                    f"Preparing to remove files...",
                    f"Deleting files...",
                    f"Verifying deletion..."
                ]
                self.show_loading_animation(f"Clearing {directory} directory", duration=2, log_messages=delete_logs)

                # Delete files
                for file_path in files:
                    try:
                        os.remove(file_path)
                        files_deleted += 1
                    except Exception as e:
                        self.print_error(f"Error deleting {file_path}: {e}")
            else:
                self.print_warning(f"Directory '{directory}' not found.")

        # Show summary
        if files_deleted > 0:
            self.print_success(f"\nSuccessfully deleted {files_deleted} files.")
        else:
            self.print_info("No files were deleted.")

        # Reset completed functions if relevant directories were cleared
        if 'data' in directories_to_clear or 'results' in directories_to_clear:
            self.completed_functions['fetch_data'] = False
            self.completed_functions['analyze_data'] = False

        if 'models' in directories_to_clear:
            self.completed_functions['train_model'] = False

        if 'results' in directories_to_clear:
            self.completed_functions['make_predictions'] = False
            self.completed_functions['backtest_strategy'] = False
            self.completed_functions['llm_analysis'] = False

        # Update available models list if models directory was cleared
        if 'models' in directories_to_clear:
            self.refresh_available_models()

        self.wait_for_key()

    def show_exit_screen(self):
        """Show the exit screen with donation information."""
        self.clear_screen()
        width = 80
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        title = "Thank you for using 3lacks Scanner"
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()
        print(f"{Fore.YELLOW}Donations accepted but not required{Style.RESET_ALL}")
        print(f"{Fore.CYAN}CGUDnm2vjTthuuxdYv7wJG6r9akxq8ascgsCXB7Dvgjz{Style.RESET_ALL}")
        print()

        # Display a countdown for 10 seconds
        for i in range(10, 0, -1):
            print(f"Exiting in {i} seconds...\r", end="")
            time.sleep(1)

        print("\nPress Enter to exit...")
        input()

    def run(self):
        """Run the terminal UI."""
        while self.running:
            self.current_menu()


def main():
    """Main function to run the terminal UI."""
    ui = TerminalUI()
    ui.run()


if __name__ == "__main__":
    main()
