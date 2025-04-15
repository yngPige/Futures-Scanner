"""
Terminal UI Module for Crypto Futures Scanner

This module provides a terminal-based user interface for the Crypto Futures Scanner application.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from colorama import init, Fore, Back, Style
from src.ui.terminal_output import TerminalOutputGenerator
from src.config import get_pybloat_path, pybloat_file_exists, PYBLOAT_DIR

# No keyboard module needed anymore

# Initialize colorama
init()

# Configure logging using custom logging utility
from src.utils.logging_utils import configure_logging

# Configure logging to only show errors in console and save to error log file
configure_logging()
logger = logging.getLogger(__name__)




class TerminalUI:
    """Terminal-based user interface for Crypto Futures Scanner."""

    def _import_from_pybloat(self):
        """Import main module from PyBloat directory."""
        import sys
        # Add PyBloat directory to sys.path if not already there
        if PYBLOAT_DIR not in sys.path:
            sys.path.insert(0, PYBLOAT_DIR)

    def __init__(self):
        """Initialize the Terminal UI."""
        self.running = True
        self.current_menu = self.main_menu

        # Default settings
        self.settings = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'limit': 500,
            'exchange': 'kraken',  # Changed default to kraken
            'model_type': 'random_forest',
            'model_path': None,
            'theme': 'dark',
            'save': True,
            'tune': False,
            'use_llm': True,  # LLM Analysis enabled by default
            'llm_model': 'phi2',
            'use_gpu': True   # GPU Acceleration enabled by default
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

        # Initialize the LLM model at startup if LLM analysis is enabled
        self.llm_analyzer = None
        if self.settings['use_llm']:
            self.preload_llm_model()

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

        self.available_exchanges = [
            'CCXT:ALL', 'kraken', 'kucoin', 'huobi'
        ]

        # Load available models
        self.refresh_available_models()

    def preload_llm_model(self):
        """Preload the LLM model at startup."""
        try:
            # Import required modules
            self._import_from_pybloat()
            from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS, DEFAULT_MODEL_PATH, POSSIBLE_MODEL_PATHS
            import os

            # Get model info
            model_info = AVAILABLE_MODELS.get(self.settings['llm_model'], {})
            model_name = model_info.get('name', self.settings['llm_model'])
            model_size = model_info.get('size_gb', 'unknown')

            # Define loading animation logs
            loading_logs = [
                "Checking model availability...",
                "Searching for models in system...",
                "Preparing model environment...",
                "Loading model weights...",
                "Initializing neural network...",
                "Setting up inference engine...",
                "Optimizing memory usage...",
                "Configuring model parameters...",
                "Preparing model context...",
                "Model ready for inference..."
            ]

            # Configure GPU usage
            n_gpu_layers = 0
            if self.settings['use_gpu']:
                print(f"{Fore.BLUE}GPU acceleration enabled for LLM inference.{Style.RESET_ALL}\n")
                n_gpu_layers = -1  # Use all layers on GPU

            # Show loading animation
            self.show_loading_animation("Loading LLM model", duration=3, log_messages=loading_logs)

            # Initialize the analyzer
            self.llm_analyzer = LocalLLMAnalyzer(
                model_name=model_name,
                n_gpu_layers=n_gpu_layers
            )

            # Check if model was loaded successfully
            if self.llm_analyzer.llm is not None:
                self.print_success("\nLLM model loaded successfully and ready for analysis!")
            else:
                self.print_warning("\nLLM model could not be loaded. LLM analysis will be disabled.")

                # Check if model exists in any of the possible paths
                model_found = False
                for path in POSSIBLE_MODEL_PATHS:
                    model_path = os.path.join(path, model_name)
                    if os.path.exists(model_path):
                        model_found = True
                        self.print_info(f"Found model at: {model_path}")
                        break

                if not model_found:
                    self.print_info("\nNo LLM model found in any of the expected locations.")
                    self.print_info(f"Please download the model '{model_name}' and place it in one of these directories:")
                    for path in POSSIBLE_MODEL_PATHS:
                        self.print_info(f"- {path}")
                else:
                    self.print_info("\nPossible reasons for loading failure:")
                    self.print_info("1. The model file might be corrupted or in an incompatible format")
                    self.print_info("2. Your system might not have enough memory to load the model")
                    self.print_info("3. The llama-cpp-python library might not be properly installed")

                self.settings['use_llm'] = False

        except Exception as e:
            self.print_error(f"Error loading LLM model: {e}")
            self.print_warning("LLM analysis will be disabled.")
            self.settings['use_llm'] = False

    def refresh_available_models(self):
        """Refresh the list of available models."""
        self.available_models = []

        # Check for models in the current directory
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.joblib') and not file.endswith('_scaler.joblib') and not file.endswith('_features.joblib'):
                    self.available_models.append(os.path.join('models', file))

        # Check for models in the PyBloat directory
        pybloat_models_dir = os.path.join(PYBLOAT_DIR, 'models')
        if os.path.exists(pybloat_models_dir):
            for file in os.listdir(pybloat_models_dir):
                if file.endswith('.joblib') and not file.endswith('_scaler.joblib') and not file.endswith('_features.joblib'):
                    model_path = os.path.join(pybloat_models_dir, file)
                    if model_path not in self.available_models:
                        self.available_models.append(model_path)

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

    # Toggle key handling is now done directly in the settings menu

    def get_input(self, prompt):
        """Get input from the user with the given prompt."""
        # Get input from user
        user_input = input(Fore.GREEN + prompt + Style.RESET_ALL)
        return user_input

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

    def show_loading_animation(self, message, duration=3, width=50, log_messages=None, compact_completion=False):
        """Show an enhanced loading bar animation with cool logs.

        Args:
            message (str): Message to display before the loading bar
            duration (int): Duration of the animation in seconds
            width (int): Width of the loading bar
            log_messages (list, optional): List of log messages to display during animation
            compact_completion (bool): Whether to use compact spacing after completion
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
        log_display_height = 8  # Increased number of log lines to show at once
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

        # Clear the log display area - use fewer lines if compact_completion is True
        clear_lines = 1 if compact_completion else log_display_height
        for _ in range(clear_lines):
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

    # Toggle indicators are now shown directly in the settings menu

    def main_menu(self):
        """Display the main menu."""
        menu_items = [
            ("1", "Fetch Data"),
            ("2", "Analyze Data"),
            ("3", "Train Model"),
            ("4", "Make Predictions"),
            ("5", "Backtest Strategy"),
            ("6", "Run All"),
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
            ("8", "Change LLM Model"),
            ("r", "Remove All LLM Models"),
            ("g", "Toggle GPU Acceleration"),
            ("s", "Toggle Save Results"),
            ("t", "Toggle Hyperparameter Tuning"),
            ("l", "Toggle LLM Analysis"),
            ("b", "Back to Main Menu")
        ]

        # Update toggle options text with current state
        for i, (key, _) in enumerate(menu_items):
            if key == 'g':
                status = f"{Fore.GREEN}ON{Style.RESET_ALL}" if self.settings['use_gpu'] else "OFF"
                menu_items[i] = (key, f"Toggle GPU Acceleration [{status}]")
            elif key == 's':
                status = f"{Fore.GREEN}ON{Style.RESET_ALL}" if self.settings['save'] else "OFF"
                menu_items[i] = (key, f"Toggle Save Results [{status}]")
            elif key == 't':
                status = f"{Fore.GREEN}ON{Style.RESET_ALL}" if self.settings['tune'] else "OFF"
                menu_items[i] = (key, f"Toggle Hyperparameter Tuning [{status}]")
            elif key == 'l':
                status = f"{Fore.GREEN}ON{Style.RESET_ALL}" if self.settings['use_llm'] else "OFF"
                menu_items[i] = (key, f"Toggle LLM Analysis [{status}]")

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
            self.change_llm_model()
        elif choice.lower() == 'g':
            self.settings['use_gpu'] = not self.settings['use_gpu']
            self.print_success(f"GPU Acceleration {'enabled' if self.settings['use_gpu'] else 'disabled'}.")
            time.sleep(1)
        elif choice.lower() == 's':
            self.settings['save'] = not self.settings['save']
            self.print_success(f"Save Results {'enabled' if self.settings['save'] else 'disabled'}.")
            time.sleep(1)
        elif choice.lower() == 't':
            # Show explanation and confirmation for hyperparameter tuning
            if not self.settings['tune']:
                self.print_header("Enable Hyperparameter Tuning?")
                print("\nHyperparameter tuning automatically tests multiple model configurations")
                print("to find the optimal settings for better prediction accuracy.")
                print("\nBenefits:")
                print("- Improved model accuracy and performance")
                print("- Better prediction quality")
                print("- Reduced overfitting")
                print("\nNote: This process takes significantly longer to train models.")

                confirm = input("\nEnable hyperparameter tuning? (y/n): ").lower().strip()
                if confirm == 'y':
                    self.settings['tune'] = True
                    self.print_success("Hyperparameter Tuning enabled.")
                else:
                    self.print_info("Hyperparameter Tuning remains disabled.")
            else:
                self.settings['tune'] = False
                self.print_success("Hyperparameter Tuning disabled.")
            time.sleep(1)
        elif choice.lower() == 'l':
            self.settings['use_llm'] = not self.settings['use_llm']
            self.print_success(f"LLM Analysis {'enabled' if self.settings['use_llm'] else 'disabled'}.")
            time.sleep(1)
        elif choice.lower() == 'r':
            self.remove_all_llm_models()
        elif choice.lower() == 'b':
            self.current_menu = self.main_menu
        else:
            self.print_error("Invalid choice. Please try again.")
            time.sleep(1)

    def change_symbol(self):
        """Change the symbol setting using the symbol selector popup."""
        self.print_header("Change Symbol")
        self.print_info("Opening symbol selector...")

        # Import the symbol selector
        from src.ui.symbol_selector import select_symbol

        # Show the symbol selector popup
        symbol, exchange, cancelled = select_symbol(
            current_symbol=self.settings['symbol'],
            current_exchange=self.settings['exchange']
        )

        # If the selection was not cancelled, update the settings
        if not cancelled and symbol:
            # Store the current exchange for comparison
            current_exchange = self.settings['exchange']

            # Update settings
            self.settings['symbol'] = symbol

            # If the exchange was changed, update it too
            if exchange != current_exchange:
                self.settings['exchange'] = exchange
                self.print_success(f"Symbol changed to {symbol} and exchange changed to {exchange}.")
            else:
                self.print_success(f"Symbol changed to {symbol}.")
        else:
            self.print_info("Symbol selection cancelled.")

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
        for i, exchange in enumerate(self.available_exchanges, 1):
            print(f"{i}. {exchange}")

        print(f"\nCurrent exchange: {self.settings['exchange']}")

        choice = self.get_input("\nEnter exchange number or custom exchange (or 's' to use symbol selector, 'b' to go back): ")

        if choice.lower() == 'b':
            return
        elif choice.lower() == 's':
            # Use the symbol selector to change both symbol and exchange
            self.print_info("Opening symbol selector...")

            # Import the symbol selector
            from src.ui.symbol_selector import select_symbol

            # Show the symbol selector popup
            symbol, exchange, cancelled = select_symbol(
                current_symbol=self.settings['symbol'],
                current_exchange=self.settings['exchange']
            )

            # If the selection was not cancelled, update the settings
            if not cancelled:
                # Update exchange
                if exchange != self.settings['exchange']:
                    self.settings['exchange'] = exchange
                    self.print_success(f"Exchange changed to {exchange}.")

                # Update symbol if it was changed
                if symbol and symbol != self.settings['symbol']:
                    self.settings['symbol'] = symbol
                    self.print_success(f"Symbol changed to {symbol}.")
            else:
                self.print_info("Selection cancelled.")
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(self.available_exchanges):
                    self.settings['exchange'] = self.available_exchanges[index]
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

        # Model type descriptions
        model_descriptions = {
            'random_forest': "Ensemble of decision trees; robust to overfitting; handles non-linear patterns well.",
            'gradient_boosting': "Sequential ensemble that corrects previous errors; higher accuracy but more prone to overfitting."
        }

        print("Available model types:")
        for i, model_type in enumerate(self.available_model_types, 1):
            description = model_descriptions.get(model_type, "")
            print(f"{i}. {model_type} - {description}")

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
        print(f"{Fore.YELLOW}License:{Style.RESET_ALL} {model_info.get('license', 'Unknown')}")
        print(f"{Fore.YELLOW}Trading Focus:{Style.RESET_ALL} {model_info.get('trading_focus', 'General purpose')}")

        # Detailed information
        print("\n" + "-" * width)
        print(f"{Fore.CYAN}COMPATIBILITY INFORMATION{Style.RESET_ALL}".center(width))
        print("-" * width)
        print(f"{Fore.YELLOW}Format:{Style.RESET_ALL} GGUF (Compatible with llama-cpp-python)")
        print(f"{Fore.YELLOW}Source:{Style.RESET_ALL} {model_info.get('url', 'Unknown')}")

        # Hardware requirements
        print("\n" + "-" * width)
        print(f"{Fore.CYAN}HARDWARE REQUIREMENTS{Style.RESET_ALL}".center(width))
        print("-" * width)
        print(f"{Fore.YELLOW}Minimum RAM:{Style.RESET_ALL} {model_info['size_gb'] * 2:.1f} GB recommended")
        print(f"{Fore.YELLOW}GPU Acceleration:{Style.RESET_ALL} {'Recommended' if model_info['size_gb'] > 1.5 else 'Optional'}")

        # Usage tips
        print("\n" + "-" * width)
        print(f"{Fore.CYAN}USAGE TIPS{Style.RESET_ALL}".center(width))
        print("-" * width)
        print(f"{Fore.YELLOW}Best for:{Style.RESET_ALL} {model_info.get('trading_focus', 'General analysis')}")
        print(f"{Fore.YELLOW}Tip:{Style.RESET_ALL} {'Enable GPU acceleration for faster inference' if model_info['size_gb'] > 1.5 else 'This model runs well on CPU-only systems'}")

        print("=" * width)

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
            from src.analysis.local_llm import check_dependencies, install_dependencies

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

    def remove_all_llm_models(self):
        """Remove all LLM models from the cache directory."""
        self.print_header("Remove All LLM Models")

        try:
            # Import required modules
            self._import_from_pybloat()
            from src.analysis.local_llm import POSSIBLE_MODEL_PATHS
            import os
            import glob
            import shutil

            # Check all possible model paths
            all_model_files = []
            for model_path in POSSIBLE_MODEL_PATHS:
                if os.path.exists(model_path):
                    # Get a list of all model files in this path
                    path_files = glob.glob(os.path.join(model_path, "*.gguf"))
                    path_files += glob.glob(os.path.join(model_path, "*.ggml"))
                    path_files += glob.glob(os.path.join(model_path, "*.bin"))

                    if path_files:
                        self.print_info(f"Found models in: {model_path}")
                        all_model_files.extend(path_files)

            if not all_model_files:
                self.print_info("No model files found in any location.")
                self.wait_for_key()
                return

            # Display the list of models to be removed
            self.print_info(f"Found {len(all_model_files)} model files:")
            for i, model_file in enumerate(all_model_files, 1):
                file_size_gb = os.path.getsize(model_file) / (1024 * 1024 * 1024)
                file_name = os.path.basename(model_file)
                file_path = os.path.dirname(model_file)
                self.print_info(f"{i}. {file_name} ({file_size_gb:.2f} GB) in {file_path}")

            # Ask for confirmation
            confirm = self.get_input("\nAre you sure you want to remove ALL LLM models? This cannot be undone. (y/n): ")
            if confirm.lower() != 'y':
                self.print_info("Operation cancelled.")
                self.wait_for_key()
                return

            # Show a second confirmation for safety
            confirm2 = self.get_input("\nFinal confirmation - remove ALL models? (type 'CONFIRM' to proceed): ")
            if confirm2 != 'CONFIRM':
                self.print_info("Operation cancelled.")
                self.wait_for_key()
                return

            # Remove all model files
            removed_count = 0
            failed_count = 0

            # Show loading animation while removing files
            delete_logs = [
                f"Preparing to remove {len(all_model_files)} model files...",
                "Checking file permissions...",
                "Removing model files...",
                "Cleaning up cache...",
                "Verifying deletion..."
            ]
            self.show_loading_animation("Removing LLM models", duration=2, log_messages=delete_logs)

            for model_file in all_model_files:
                try:
                    os.remove(model_file)
                    removed_count += 1
                except Exception as e:
                    self.print_error(f"Failed to remove {os.path.basename(model_file)}: {e}")
                    failed_count += 1

            # Reset the LLM analyzer
            self.llm_analyzer = None

            # Display results
            if removed_count > 0:
                self.print_success(f"Successfully removed {removed_count} model files.")
            if failed_count > 0:
                self.print_warning(f"Failed to remove {failed_count} model files.")

            self.wait_for_key()

        except Exception as e:
            self.print_error(f"Error removing LLM models: {e}")
            self.wait_for_key()



    def change_llm_model(self):
        """Change the LLM model setting."""
        self.print_header("Change LLM Model")

        # Import available models from local_llm module
        try:
            from src.analysis.local_llm import AVAILABLE_MODELS

            # Available LLM models
            available_models = list(AVAILABLE_MODELS.keys())

            # Display all available open-source models
            print(f"{Fore.CYAN}Available Open-Source LLM Models:{Style.RESET_ALL}")
            for i, model_key in enumerate(available_models, 1):
                model_info = AVAILABLE_MODELS[model_key]
                focus = model_info.get('trading_focus', '')
                license_info = model_info.get('license', 'Unknown License')
                license_color = Fore.GREEN if 'MIT' in license_info or 'Apache' in license_info else Fore.YELLOW
                print(f"{i}. {Fore.WHITE}{model_key}{Style.RESET_ALL} - {model_info['description']} ({model_info['size_gb']:.2f} GB)")
                print(f"   {Fore.BLUE}Focus:{Style.RESET_ALL} {focus} | {license_color}License: {license_info}{Style.RESET_ALL}")

            # Use all models for selection
            all_models = available_models

            print(f"\nCurrent LLM model: {Fore.YELLOW}{self.settings['llm_model']}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}Options:{Style.RESET_ALL}")
            print(f"- Enter a model number to view details and select a model")
            print(f"- Enter '{Fore.GREEN}b{Style.RESET_ALL}' to go back")

            choice = self.get_input("\nEnter your choice: ")

            if choice.lower() == 'b':
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
                        self.print_info(f"Note: You need to manually download this model ({model_info['size_gb']:.2f} GB) and place it in the models folder.")
                        self.print_info(f"Model filename: {model_info.get('name')}")
                        self.print_info(f"Download URL: {model_info.get('url')}")

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

        # Add terminal chart option (default to False)
        args.terminal_chart = False

        # Add no_display option (default to False)
        args.no_display = False

        return args

    def fetch_data(self):
        """Fetch data based on current settings."""
        self.print_header("Fetching Data")

        try:
            # Import main from PyBloat directory
            self._import_from_pybloat()
            from main import fetch_data

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2, compact_completion=True)

            df = fetch_data(args)

            if df is not None:
                self.print_success(f"Successfully fetched {len(df)} rows of data.")
                self.print_info(f"Time range: {df.index.min()} to {df.index.max()}")
                self.print_info(f"Latest price: {df['close'].iloc[-1]:.5f}")

                # Display which exchange was actually used (for CCXT:ALL mode)
                if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                    self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

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
            # Import main from PyBloat directory
            self._import_from_pybloat()
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
                # Display which exchange was actually used (for CCXT:ALL mode)
                if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                    self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

                # Mark fetch_data as completed
                self.completed_functions['fetch_data'] = True

                self.print_info("Performing technical analysis...")

                # Show loading animation with detailed technical analysis logs
                analysis_logs = [
                    "Initializing technical analysis engine...",
                    "Standardizing price data columns...",
                    "Computing trend indicators (SMA, EMA, TEMA)...",
                    "Calculating momentum oscillators (RSI, Stochastic)...",
                    "Generating MACD signal lines and histogram...",
                    "Computing Bollinger Bands and %B indicator...",
                    "Calculating Average True Range for volatility...",
                    "Identifying support and resistance levels...",
                    "Detecting price patterns and chart formations...",
                    "Computing On-Balance Volume and Money Flow...",
                    "Calculating Ichimoku Cloud components...",
                    "Generating Fibonacci retracement levels...",
                    "Identifying divergence patterns in oscillators...",
                    "Calculating pivot points and price channels...",
                    "Generating final trading signals and indicators..."
                ]
                self.show_loading_animation("Calculating technical indicators", duration=3, log_messages=analysis_logs, compact_completion=True)

                # Analyze the data
                df_analyzed = analyze_data(df, args)

                # Add more detailed logs about the analysis results
                if df_analyzed is not None and not df_analyzed.empty:
                    # Log information about the indicators that were calculated
                    indicator_groups = {
                        "Trend": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['sma', 'ema', 'tema', 'wma', 'adx'])],
                        "Momentum": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['rsi', 'stoch', 'cci', 'mfi', 'roc'])],
                        "Volatility": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['atr', 'bb', 'kc', 'donchian'])],
                        "Volume": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['obv', 'cmf', 'vwap', 'volume'])],
                        "Oscillators": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['macd', 'ppo', 'tsi'])]
                    }

                    # Log the number of indicators in each group
                    for group, indicators in indicator_groups.items():
                        if indicators:
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            self.collected_logs.append((timestamp, f"Calculated {len(indicators)} {group.lower()} indicators"))

                    # Log information about the latest values of key indicators
                    latest = df_analyzed.iloc[-1]
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    # RSI
                    if 'rsi_14' in df_analyzed.columns:
                        rsi = latest['rsi_14']
                        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        self.collected_logs.append((timestamp, f"RSI(14): {rsi:.2f} - {rsi_status}"))

                    # MACD
                    if all(col in df_analyzed.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
                        macd = latest['MACD_12_26_9']
                        signal = latest['MACDs_12_26_9']
                        macd_status = "Bullish" if macd > signal else "Bearish"
                        self.collected_logs.append((timestamp, f"MACD: {macd:.4f} - {macd_status}"))

                    # Moving Averages
                    if 'sma_50' in df_analyzed.columns and 'sma_200' in df_analyzed.columns:
                        sma_50 = latest['sma_50']
                        sma_200 = latest['sma_200']
                        ma_status = "Golden Cross" if sma_50 > sma_200 else "Death Cross"
                        self.collected_logs.append((timestamp, f"MA Cross: {ma_status} (SMA50: {sma_50:.2f}, SMA200: {sma_200:.2f})"))

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_success(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators.")
                    # Mark analyze_data as completed
                    self.completed_functions['analyze_data'] = True

                    # Show key indicators and metrics in a combined format
                    latest = df_analyzed.iloc[-1]
                    print(f"\n{Fore.YELLOW}KEY INDICATORS & METRICS{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}---------------------{Style.RESET_ALL}")

                    # Create a two-column layout for indicators and metrics
                    left_column = []
                    right_column = []

                    # Show RSI if available
                    if 'rsi_14' in df_analyzed.columns:
                        rsi = latest['rsi_14']
                        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        rsi_color = Fore.RED if rsi < 30 else Fore.GREEN if rsi > 70 else Fore.WHITE
                        left_column.append(f"RSI (14): {rsi_color}{rsi:.2f} - {rsi_status}{Style.RESET_ALL}")

                    # Show MACD if available
                    if 'MACD_12_26_9' in df_analyzed.columns and 'MACDs_12_26_9' in df_analyzed.columns:
                        macd = latest['MACD_12_26_9']
                        signal = latest['MACDs_12_26_9']
                        macd_hist = macd - signal
                        macd_status = "Bullish" if macd > signal else "Bearish"
                        macd_color = Fore.GREEN if macd > signal else Fore.RED
                        left_column.append(f"MACD: {macd_color}{macd:.4f} - {macd_status}{Style.RESET_ALL} (H: {macd_hist:.4f})")

                    # Show Moving Averages if available
                    if 'sma_50' in df_analyzed.columns and 'sma_200' in df_analyzed.columns:
                        sma_50 = latest['sma_50']
                        sma_200 = latest['sma_200']
                        ma_status = "Golden Cross" if sma_50 > sma_200 else "Death Cross"
                        ma_color = Fore.GREEN if sma_50 > sma_200 else Fore.RED
                        left_column.append(f"MA Cross: {ma_color}{ma_status}{Style.RESET_ALL} ({sma_50:.2f}/{sma_200:.2f})")

                    # Show Bollinger Bands if available
                    if 'BBL_20_2.0' in df_analyzed.columns and 'BBM_20_2.0' in df_analyzed.columns and 'BBU_20_2.0' in df_analyzed.columns:
                        lower = latest['BBL_20_2.0']
                        middle = latest['BBM_20_2.0']
                        upper = latest['BBU_20_2.0']
                        price = latest['close']
                        bb_position = (price - lower) / (upper - lower) * 100 if (upper - lower) > 0 else 50
                        bb_status = "Lower Band" if price < lower else "Upper Band" if price > upper else "Middle Band"
                        bb_color = Fore.RED if price < lower else Fore.GREEN if price > upper else Fore.WHITE
                        left_column.append(f"BB: {bb_color}{bb_status}{Style.RESET_ALL} ({bb_position:.1f}%)")

                    # Add price information to right column
                    right_column.append(f"Price: {Fore.CYAN}{latest['close']:.5f}{Style.RESET_ALL}")
                    right_column.append(f"Volume: {Fore.YELLOW}{latest.get('volume', 'N/A'):.0f}{Style.RESET_ALL}")

                    # Add more metrics if available
                    if 'atr_14' in df_analyzed.columns:
                        right_column.append(f"ATR (14): {Fore.MAGENTA}{latest['atr_14']:.5f}{Style.RESET_ALL}")

                    if 'obv' in df_analyzed.columns:
                        right_column.append(f"OBV: {Fore.BLUE}{latest['obv']:.0f}{Style.RESET_ALL}")

                    # Ensure both columns have the same number of items
                    max_items = max(len(left_column), len(right_column))
                    left_column.extend([''] * (max_items - len(left_column)))
                    right_column.extend([''] * (max_items - len(right_column)))

                    # Combine columns with proper spacing
                    for i in range(max_items):
                        left_item = left_column[i]
                        right_item = right_column[i]
                        # Remove ANSI color codes for length calculation
                        left_item_clean = left_item.replace(Fore.RED, '').replace(Fore.GREEN, '').replace(Fore.WHITE, '').replace(Fore.YELLOW, '').replace(Fore.CYAN, '').replace(Fore.MAGENTA, '').replace(Fore.BLUE, '').replace(Style.RESET_ALL, '')
                        print(f"{left_item}{' ' * (40 - len(left_item_clean))}{right_item}")

                    print()

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
            # Import main from PyBloat directory
            self._import_from_pybloat()
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
            self.show_loading_animation("Retrieving market data", duration=2, log_messages=fetch_logs, compact_completion=True)

            df = fetch_data(args)

            if df is not None:
                # Display which exchange was actually used (for CCXT:ALL mode)
                if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                    self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

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
                self.show_loading_animation("Analyzing market data", duration=2.5, log_messages=analysis_logs, compact_completion=True)

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
                    self.show_loading_animation("Training machine learning model", duration=4, log_messages=train_logs, compact_completion=True)

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
            # Import main from PyBloat directory
            self._import_from_pybloat()
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
            self.show_loading_animation("Retrieving market data", duration=2, log_messages=fetch_logs, compact_completion=True)

            df = fetch_data(args)

            if df is not None:
                # Display which exchange was actually used (for CCXT:ALL mode)
                if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                    self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

                self.print_info("Performing technical analysis...")

                # Show loading animation while analyzing data
                self.show_loading_animation("Calculating technical indicators", duration=2.5, compact_completion=True)

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
                    self.show_loading_animation("Running prediction model", duration=2, log_messages=predict_logs, compact_completion=True)

                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_success("Successfully made predictions.")
                        # Mark make_predictions as completed
                        self.completed_functions['make_predictions'] = True

                        # Display prediction summary with improved formatting
                        print(f"\n{Fore.YELLOW}PREDICTION SUMMARY{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}------------------{Style.RESET_ALL}")

                        if 'prediction' in df_predictions.columns:
                            # Count predictions
                            pred_counts = df_predictions['prediction'].value_counts()
                            total_preds = len(df_predictions)

                            # Calculate percentages
                            bullish_count = pred_counts.get(1, 0)
                            bearish_count = pred_counts.get(0, 0)
                            bullish_pct = (bullish_count / total_preds) * 100 if total_preds > 0 else 0
                            bearish_pct = (bearish_count / total_preds) * 100 if total_preds > 0 else 0

                            # Latest prediction
                            latest_pred = df_predictions['prediction'].iloc[-1]
                            latest_prob = df_predictions['prediction_probability'].iloc[-1] if 'prediction_probability' in df_predictions.columns else None

                            pred_text = "Bullish" if latest_pred == 1 else "Bearish"
                            pred_color = Fore.GREEN if latest_pred == 1 else Fore.RED
                            prob_text = f" (Confidence: {latest_prob:.2f})" if latest_prob is not None else ""

                            # Create a two-column layout for prediction summary
                            left_column = []
                            right_column = []

                            # Add prediction info to left column
                            left_column.append(f"Prediction: {pred_color}{pred_text}{Style.RESET_ALL}{prob_text}")
                            left_column.append(f"Bullish signals: {Fore.GREEN}{bullish_count} ({bullish_pct:.1f}%){Style.RESET_ALL}")
                            left_column.append(f"Bearish signals: {Fore.RED}{bearish_count} ({bearish_pct:.1f}%){Style.RESET_ALL}")

                            # Add price info to right column
                            latest = df_predictions.iloc[-1]
                            right_column.append(f"Current price: {Fore.CYAN}{latest['close']:.5f}{Style.RESET_ALL}")

                            # Add entry/exit points with TP/SL levels if available
                            if all(col in df_predictions.columns for col in ['entry_price', 'stop_loss', 'take_profit', 'risk_reward']):
                                right_column.append(f"Entry Price: {Fore.CYAN}{latest['entry_price']:.5f}{Style.RESET_ALL}")
                                right_column.append(f"Stop Loss: {Fore.RED}{latest['stop_loss']:.5f}{Style.RESET_ALL}")
                                right_column.append(f"Take Profit: {Fore.GREEN}{latest['take_profit']:.5f}{Style.RESET_ALL}")
                                right_column.append(f"Risk/Reward: {Fore.YELLOW}1:{latest['risk_reward']:.2f}{Style.RESET_ALL}")

                            # Ensure both columns have the same number of items
                            max_items = max(len(left_column), len(right_column))
                            left_column.extend([''] * (max_items - len(left_column)))
                            right_column.extend([''] * (max_items - len(right_column)))

                            # Combine columns with proper spacing
                            for i in range(max_items):
                                left_item = left_column[i]
                                right_item = right_column[i]
                                # Remove ANSI color codes for length calculation
                                left_item_clean = left_item.replace(Fore.RED, '').replace(Fore.GREEN, '').replace(Fore.WHITE, '').replace(Fore.YELLOW, '').replace(Fore.CYAN, '').replace(Style.RESET_ALL, '')
                                print(f"{left_item}{' ' * (40 - len(left_item_clean))}{right_item}")
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
            # Import main from PyBloat directory
            self._import_from_pybloat()
            from main import fetch_data, analyze_data, predict, backtest

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time
            operation_start_time = datetime.now()

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2, compact_completion=True)

            df = fetch_data(args)

            if df is not None:
                # Display which exchange was actually used (for CCXT:ALL mode)
                if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                    self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")
                self.print_info("Performing technical analysis...")

                # Show loading animation while analyzing data
                self.show_loading_animation("Calculating technical indicators", duration=2.5, compact_completion=True)

                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info("Making predictions...")

                    # Show loading animation while making predictions
                    self.show_loading_animation("Running prediction model", duration=2, compact_completion=True)

                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_info("Backtesting predictions...")

                        # Show loading animation while backtesting
                        self.show_loading_animation("Simulating trading strategy", duration=3, compact_completion=True)

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
                        self.show_loading_animation("Preparing backtest report", duration=1.5, compact_completion=True)

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
            # Import main from PyBloat directory
            self._import_from_pybloat()
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

            # Display which exchange was actually used (for CCXT:ALL mode)
            if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

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

            # Check if model already exists and has a reasonable size (at least 1MB)
            if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 1000000:
                file_size_gb = os.path.getsize(model_file_path) / (1024 * 1024 * 1024)
                self.print_info(f"Model already exists at {model_file_path}")
                self.print_info(f"Model file size: {file_size_gb:.2f} GB")
            # If file doesn't exist or is too small (likely a failed download)
            else:
                if os.path.exists(model_file_path) and os.path.getsize(model_file_path) <= 1000000:
                    self.print_warning(f"Model file exists but appears to be corrupted or incomplete: {model_name}")
                    self.print_warning(f"File size: {os.path.getsize(model_file_path) / (1024*1024):.2f} MB")
                    # Remove the incomplete file
                    try:
                        os.remove(model_file_path)
                        self.print_info("Removed incomplete file.")
                    except Exception as e:
                        self.print_error(f"Failed to remove incomplete file: {str(e)}")
                else:
                    self.print_warning(f"Model file not found: {model_name}")

                # Show model info
                self.print_info(f"Model: {self.settings['llm_model']} ({model_info.get('size_gb', 'unknown')} GB)")
                self.print_info(f"Description: {model_info.get('description', 'No description available')}")

                # Inform user they need to download the model manually
                self.print_info("You need to download the model manually to use LLM analysis.")
                self.print_info(f"Model filename: {model_name}")
                self.print_info(f"Download URL: {model_info.get('url')}")
                self.print_info(f"Save to one of these locations:")
                from src.analysis.local_llm import POSSIBLE_MODEL_PATHS
                for path in POSSIBLE_MODEL_PATHS:
                    self.print_info(f"- {path}")
                self.print_info("\nCannot proceed with LLM analysis without the model.")
                self.wait_for_key()
                return

            # Use the preloaded LLM analyzer if available, or initialize a new one if needed
            if self.llm_analyzer is None or self.llm_analyzer.llm is None:
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
                self.show_loading_animation("Loading AI model", duration=3, log_messages=model_logs, compact_completion=True)

                # Configure GPU usage
                n_gpu_layers = 0
                if self.settings['use_gpu']:
                    self.print_info("GPU acceleration enabled. Using GPU for inference.")
                    n_gpu_layers = -1  # Use all layers on GPU

                # Initialize the analyzer
                self.llm_analyzer = LocalLLMAnalyzer(
                    model_name=model_name,
                    n_gpu_layers=n_gpu_layers
                )
            else:
                self.print_info(f"Using preloaded LLM model: {self.settings['llm_model']}")

            # Use the analyzer (preloaded or newly initialized)
            llm_analyzer = self.llm_analyzer

            # Check if LLM was initialized successfully
            if llm_analyzer.llm is None:
                self.print_warning("WARNING - Failed to initialize LLM model.")
                self.print_warning(f"Model file not found or corrupted: {model_file_path}")
                self.print_info("Using fallback technical indicator-based recommendation system instead.")
                self.print_info("\nTo use the full LLM capabilities, you can:")
                self.print_info("1. Try downloading the model again using the dedicated script:")
                self.print_info(f"   python download_llm_model.py download {self.settings['llm_model']} --force")
                self.print_info("2. Check if your system meets the requirements for running LLMs:")
                self.print_info("   - At least 16GB of RAM")
                self.print_info("   - For GPU acceleration: NVIDIA GPU with at least 8GB VRAM")
                self.print_info("3. Try a smaller model if your system has limited resources")

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
            self.show_loading_animation("AI analyzing market data", duration=3, log_messages=analysis_logs, compact_completion=True)
            recommendation = llm_analyzer.analyze(df_analyzed)

            # Check if we got a recommendation
            if recommendation is None:
                self.print_error(f"LLM analysis failed: No recommendation returned")
                self.wait_for_key()
                return

            # Check if it's an error (but not a fallback recommendation)
            if "error" in recommendation and recommendation.get("model") != "FALLBACK_TECHNICAL_INDICATORS" and recommendation.get("model") != "FALLBACK_ERROR":
                self.print_error(f"LLM analysis failed: {recommendation['error']}")
                self.wait_for_key()
                return

            # Check if it's a fallback recommendation
            if recommendation.get("model") == "FALLBACK_TECHNICAL_INDICATORS":
                self.print_info("Using fallback technical indicator-based recommendation system.")

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
            if 'recommendation' in recommendation:
                # Original LLM format
                self.print_info(f"\nTrading Recommendation: {Fore.YELLOW}{recommendation['recommendation']}{Style.RESET_ALL}")
                self.print_info(f"Risk Assessment: {Fore.YELLOW}{recommendation.get('risk', 'UNKNOWN')}{Style.RESET_ALL}")
                self.print_info(f"Model Used: {Fore.CYAN}{recommendation.get('model', model_name)}{Style.RESET_ALL}")
                self.print_info(f"Analysis Timestamp: {recommendation.get('timestamp', datetime.now().isoformat())}")
            else:
                # Fallback format
                self.print_info(f"\nTrading Signal: {Fore.YELLOW}{recommendation.get('signal', 'neutral').upper()}{Style.RESET_ALL}")
                self.print_info(f"Market Summary: {Fore.YELLOW}{recommendation.get('market_summary', 'Unknown')}{Style.RESET_ALL}")
                self.print_info(f"Confidence: {Fore.YELLOW}{recommendation.get('confidence', 'low')}{Style.RESET_ALL}")
                self.print_info(f"Generated By: {Fore.CYAN}{recommendation.get('generated_by', 'fallback_system')}{Style.RESET_ALL}")

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

            # Print full analysis if available
            if 'analysis' in recommendation:
                print(recommendation['analysis'])
            elif 'reasoning' in recommendation:
                print(f"\n{Fore.CYAN}Analysis:{Style.RESET_ALL}")
                print(recommendation['reasoning'])

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
            # Import main from PyBloat directory
            self._import_from_pybloat()
            from main import fetch_data, analyze_data, train_model, predict, backtest

            args = self.build_command_args()

            # Clear any previous logs
            self.collected_logs = []

            # Record operation start time

            # Fetch data
            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2, compact_completion=True)

            df = fetch_data(args)
            if df is None:
                self.print_error("Failed to fetch data.")
                self.wait_for_key()
                return

            # Display which exchange was actually used (for CCXT:ALL mode)
            if 'exchange' in df.attrs and df.attrs['exchange'] != self.settings['exchange']:
                self.print_info(f"Data fetched from {df.attrs['exchange']} exchange")

            self.print_success(f"Successfully fetched {len(df)} rows of data.")
            self.print_info(f"Time range: {df.index.min()} to {df.index.max()}")

            # Analyze data
            self.print_info("Performing technical analysis...")

            # Show enhanced loading animation with detailed technical analysis logs
            analysis_logs = [
                "Initializing technical analysis engine...",
                "Standardizing price data columns...",
                "Computing trend indicators (SMA, EMA, TEMA)...",
                "Calculating momentum oscillators (RSI, Stochastic)...",
                "Generating MACD signal lines and histogram...",
                "Computing Bollinger Bands and %B indicator...",
                "Calculating Average True Range for volatility...",
                "Identifying support and resistance levels...",
                "Detecting price patterns and chart formations...",
                "Computing On-Balance Volume and Money Flow...",
                "Calculating Ichimoku Cloud components...",
                "Generating Fibonacci retracement levels...",
                "Identifying divergence patterns in oscillators...",
                "Calculating pivot points and price channels...",
                "Generating final trading signals and indicators..."
            ]
            self.show_loading_animation("Calculating technical indicators", duration=3, log_messages=analysis_logs, compact_completion=True)

            # Analyze the data
            df_analyzed = analyze_data(df, args)

            # Add more detailed logs about the analysis results
            if df_analyzed is not None and not df_analyzed.empty:
                # Log information about the indicators that were calculated
                indicator_groups = {
                    "Trend": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['sma', 'ema', 'tema', 'wma', 'adx'])],
                    "Momentum": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['rsi', 'stoch', 'cci', 'mfi', 'roc'])],
                    "Volatility": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['atr', 'bb', 'kc', 'donchian'])],
                    "Volume": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['obv', 'cmf', 'vwap', 'volume'])],
                    "Oscillators": [col for col in df_analyzed.columns if any(x in col.lower() for x in ['macd', 'ppo', 'tsi'])]
                }

                # Log the number of indicators in each group
                for group, indicators in indicator_groups.items():
                    if indicators:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.collected_logs.append((timestamp, f"Calculated {len(indicators)} {group.lower()} indicators"))
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
            self.show_loading_animation("Training machine learning model", duration=4, log_messages=train_logs, compact_completion=True)

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
            self.show_loading_animation("Running prediction model", duration=2, compact_completion=True)

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
            self.show_loading_animation("Simulating trading strategy", duration=3, compact_completion=True)

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

            # Display a compact summary instead of the full analysis
            self.print_info("\nGenerating compact analysis summary...")

            # Show loading animation while preparing output
            self.show_loading_animation("Preparing analysis summary", duration=1, compact_completion=True)

            # Create a compact summary of the analysis
            self.print_success("Analysis Summary:")

            # Print a compact summary of the key metrics and signals
            self._display_compact_summary(df_predictions, args.symbol, args.timeframe,
                                        performance_metrics, trading_metrics)

            # Perform LLM analysis if enabled
            if args.use_llm:
                self.print_info("\nPerforming LLM analysis...")

                # Show loading animation for LLM analysis
                self.show_loading_animation("Initializing AI analysis", duration=2, compact_completion=True)

                from src.analysis.local_llm import LocalLLMAnalyzer, AVAILABLE_MODELS

                # Get model info
                model_info = AVAILABLE_MODELS.get(args.llm_model, {})
                model_name = model_info.get('name', args.llm_model)

                # Configure GPU usage
                n_gpu_layers = 0
                if args.use_gpu:
                    self.print_info("GPU acceleration enabled for LLM inference.")
                    n_gpu_layers = -1  # Use all layers on GPU

                # Use the preloaded LLM analyzer if available, or initialize a new one if needed
                if self.llm_analyzer is None or self.llm_analyzer.llm is None:
                    # Initialize the analyzer
                    self.llm_analyzer = LocalLLMAnalyzer(
                        model_name=model_name,
                        n_gpu_layers=n_gpu_layers
                    )
                else:
                    self.print_info(f"Using preloaded LLM model: {self.settings['llm_model']}")

                # Use the analyzer (preloaded or newly initialized)
                llm_analyzer = self.llm_analyzer

                # Show loading animation for LLM processing
                self.show_loading_animation("AI analyzing market data", duration=3, compact_completion=True)

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
                            print(f"{Fore.GREEN}Entry Price:{Style.RESET_ALL} {entry_price:.5f}")

                        if stop_loss is not None:
                            print(f"{Fore.RED}Stop Loss:{Style.RESET_ALL} {stop_loss:.5f}")

                        if take_profit is not None:
                            print(f"{Fore.GREEN}Take Profit:{Style.RESET_ALL} {take_profit:.5f}")

                        if risk_reward is not None:
                            print(f"{Fore.YELLOW}Risk/Reward Ratio:{Style.RESET_ALL} {risk_reward}")

                    # Print full analysis
                    print("\n" + recommendation['analysis'])

                    # Ask user if they want to display a chart
                    chart_choice = self.get_input("\nDisplay chart? (t)erminal, (b)rowser, or (n)o: ").lower()

                    if chart_choice == 't':
                        # Set terminal chart option
                        args.terminal_chart = True
                        args.interactive = False
                        self.print_info("Displaying chart in terminal...")
                        # Import main from PyBloat directory
                        self._import_from_pybloat()
                        from main import visualize
                        visualize(df_analyzed, args, llm_recommendation=recommendation)
                    elif chart_choice == 'b':
                        # Use browser chart
                        args.terminal_chart = False
                        args.interactive = True
                        self.print_info("Displaying chart in browser...")
                        # Import main from PyBloat directory
                        self._import_from_pybloat()
                        from main import visualize
                        visualize(df_analyzed, args, llm_recommendation=recommendation)
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
        """Display collected logs in a comprehensive format.

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

        # Define log groups based on keywords for better organization
        group_keywords = {
            'data_retrieval': ['data', 'candles', 'downloading', 'fetching', 'retrieving', 'connecting', 'exchange'],
            'data_processing': ['processing', 'validating', 'formatting', 'checking', 'preparing dataset'],
            'technical_analysis': ['analyzing', 'indicators', 'technical', 'calculating', 'computing'],
            'pattern_detection': ['patterns', 'support/resistance', 'trend', 'crossover', 'divergence'],
            'indicator_calculation': ['moving averages', 'rsi', 'macd', 'bollinger', 'momentum', 'oscillators'],
            'model_training': ['model', 'machine learning', 'training', 'hyperparameter'],
            'prediction': ['prediction', 'signals', 'forecasting', 'probabilities'],
            'llm_analysis': ['llm', 'ai', 'neural network', 'analyzing market data'],
            'completion': ['completed', 'success', 'finished', '✓']
        }

        # Function to determine log group
        def get_log_group(message):
            message_lower = message.lower()
            for group, keywords in group_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    return group
            return 'other'

        # Group logs by category
        grouped_logs = {}
        for timestamp, message in filtered_logs:
            group = get_log_group(message)
            if group not in grouped_logs:
                grouped_logs[group] = []
            grouped_logs[group].append((timestamp, message))

        # Define the order of groups for display
        group_order = [
            'data_retrieval',
            'data_processing',
            'technical_analysis',
            'indicator_calculation',
            'pattern_detection',
            'model_training',
            'prediction',
            'llm_analysis',
            'completion',
            'other'
        ]

        # Print section headers and logs by group
        for group in group_order:
            if group in grouped_logs and grouped_logs[group]:
                # Print section header
                group_display_name = group.replace('_', ' ').title()
                print(f"\n{Fore.YELLOW}{group_display_name}:{Style.RESET_ALL}")

                # Sort logs within group by timestamp
                group_logs = sorted(grouped_logs[group], key=lambda x: x[0])

                # Print logs with colors
                for timestamp, message in group_logs:
                    # Choose color based on message content
                    if "error" in message.lower() or "failed" in message.lower():
                        color = Fore.RED
                    elif "warning" in message.lower():
                        color = Fore.YELLOW
                    elif "completed" in message.lower() or "success" in message.lower() or "✓" in message:
                        color = Fore.GREEN
                    else:
                        # Use different colors for different groups
                        if group == 'data_retrieval':
                            color = Fore.CYAN
                        elif group == 'technical_analysis' or group == 'indicator_calculation':
                            color = Fore.MAGENTA
                        elif group == 'pattern_detection':
                            color = Fore.BLUE
                        elif group == 'model_training' or group == 'prediction':
                            color = Fore.GREEN
                        elif group == 'llm_analysis':
                            color = Fore.YELLOW
                        else:
                            color = Fore.WHITE

                    # Print formatted log entry with timestamp
                    print(f"  {Fore.WHITE}[{timestamp}]{Style.RESET_ALL} {color}{message}{Style.RESET_ALL}")

        # Add a summary section if we have completion logs
        if 'completion' in grouped_logs and grouped_logs['completion']:
            print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}")
            for timestamp, message in sorted(grouped_logs['completion'], key=lambda x: x[0]):
                print(f"  {Fore.WHITE}[{timestamp}]{Style.RESET_ALL} {Fore.GREEN}{message}{Style.RESET_ALL}")

        # Clear logs after displaying them
        self.collected_logs = []

        print("=" * width)
        print()

    def show_how_to_use(self):
        """Display the 'How to Use' guide."""
        self.clear_screen()
        width = 80
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        title = "3lacks Scanner - How to Use"
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()

        # List of README files
        readme_files = [
            "README.md",
            "README_EXECUTABLE.md",
            "TERMINAL_UI_GUIDE.md",
            "tests/README.md"
        ]

        # Display message to read README files
        print(f"{Fore.YELLOW}Please read the README markdown files for detailed instructions:{Style.RESET_ALL}\n")

        for i, file in enumerate(readme_files, 1):
            if os.path.exists(file):
                print(f"{Fore.GREEN}{i}. {file}{Style.RESET_ALL} - {self._get_readme_title(file)}")
            else:
                print(f"{Fore.RED}{i}. {file} (not found){Style.RESET_ALL}")

        print()
        print(f"{Fore.YELLOW}Additional documentation:{Style.RESET_ALL}")
        print(f"{Fore.BLUE}• docs/how_to_use.md{Style.RESET_ALL} - Detailed user guide")
        print()
        print(f"{Fore.CYAN}The README files contain comprehensive information about:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}• Installation and setup{Style.RESET_ALL}")
        print(f"{Fore.WHITE}• Features and functionality{Style.RESET_ALL}")
        print(f"{Fore.WHITE}• Usage instructions{Style.RESET_ALL}")
        print(f"{Fore.WHITE}• Technical details{Style.RESET_ALL}")
        print(f"{Fore.WHITE}• Testing procedures{Style.RESET_ALL}")
        print()
        print(f"{Fore.MAGENTA}You can open these files in any text editor or markdown viewer.{Style.RESET_ALL}")

        self.wait_for_key()

    def _display_compact_summary(self, df, symbol, timeframe, performance_metrics=None, trading_metrics=None):
        """Display a compact summary of the analysis results.

        Args:
            df (pd.DataFrame): DataFrame with analysis data
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the analysis
            performance_metrics (dict, optional): Performance metrics from backtest
            trading_metrics (dict, optional): Trading metrics from backtest
        """
        try:
            # Get the latest data point
            latest = df.iloc[-1]

            # Create a compact box for the summary
            width = 80
            print("\n" + "=" * width)
            print(f"{Fore.CYAN}ANALYSIS SUMMARY - {symbol} ({timeframe}){Style.RESET_ALL}".center(width))
            print("=" * width)

            # Market overview in a single line
            print(f"{Fore.YELLOW}MARKET:{Style.RESET_ALL} Open: {latest.get('open', 'N/A'):.5f} | High: {latest.get('high', 'N/A'):.5f} | Low: {latest.get('low', 'N/A'):.5f} | Close: {latest.get('close', 'N/A'):.5f} | Vol: {latest.get('volume', 'N/A'):.0f}")

            # Key indicators in a compact format
            indicators = []

            # Add RSI
            if 'rsi_14' in df.columns:
                rsi = latest['rsi_14']
                rsi_color = Fore.RED if rsi < 30 else Fore.GREEN if rsi > 70 else Fore.WHITE
                indicators.append(f"RSI: {rsi_color}{rsi:.2f}{Style.RESET_ALL}")

            # Add MACD
            if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
                macd = latest['MACD_12_26_9']
                signal = latest['MACDs_12_26_9']
                macd_color = Fore.GREEN if macd > signal else Fore.RED
                indicators.append(f"MACD: {macd_color}{macd:.4f}{Style.RESET_ALL}")

            # Add MA Cross
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                sma_50 = latest['sma_50']
                sma_200 = latest['sma_200']
                ma_color = Fore.GREEN if sma_50 > sma_200 else Fore.RED
                indicators.append(f"MA: {ma_color}{sma_50:.2f}/{sma_200:.2f}{Style.RESET_ALL}")

            # Print indicators in a single line
            print(f"{Fore.YELLOW}INDICATORS:{Style.RESET_ALL} {' | '.join(indicators)}")

            # Signal and prediction in a single line
            signals = []

            # Add overall signal
            if 'signal' in df.columns:
                signal_value = latest['signal']
                if signal_value > 0.6:
                    signal_text = "Strong Buy"
                    signal_color = Fore.GREEN
                elif signal_value > 0:
                    signal_text = "Buy"
                    signal_color = Fore.GREEN
                elif signal_value < -0.6:
                    signal_text = "Strong Sell"
                    signal_color = Fore.RED
                elif signal_value < 0:
                    signal_text = "Sell"
                    signal_color = Fore.RED
                else:
                    signal_text = "Neutral"
                    signal_color = Fore.WHITE
                signals.append(f"Signal: {signal_color}{signal_text}{Style.RESET_ALL}")

            # Add prediction
            if 'prediction' in df.columns:
                pred_value = latest['prediction']
                pred_prob = latest.get('prediction_probability', 0.5)
                pred_text = "Bullish" if pred_value == 1 else "Bearish"
                pred_color = Fore.GREEN if pred_value == 1 else Fore.RED
                signals.append(f"Prediction: {pred_color}{pred_text}{Style.RESET_ALL} ({pred_prob:.2f})")

            # Print signals in a single line
            if signals:
                print(f"{Fore.YELLOW}SIGNALS:{Style.RESET_ALL} {' | '.join(signals)}")

            # Trading levels in a single line
            if all(col in df.columns for col in ['entry_price', 'stop_loss', 'take_profit', 'risk_reward']):
                print(f"{Fore.YELLOW}LEVELS:{Style.RESET_ALL} Entry: {Fore.CYAN}{latest['entry_price']:.5f}{Style.RESET_ALL} | SL: {Fore.RED}{latest['stop_loss']:.5f}{Style.RESET_ALL} | TP: {Fore.GREEN}{latest['take_profit']:.5f}{Style.RESET_ALL} | R/R: 1:{latest['risk_reward']:.2f}")

            # Performance metrics in a compact format
            if performance_metrics or trading_metrics:
                # Combine the most important metrics
                key_metrics = []

                # Add key performance metrics
                if performance_metrics:
                    if 'accuracy' in performance_metrics:
                        key_metrics.append(f"Accuracy: {performance_metrics['accuracy']:.4f}")
                    if 'f1_score' in performance_metrics:
                        key_metrics.append(f"F1: {performance_metrics['f1_score']:.4f}")

                # Add key trading metrics
                if trading_metrics:
                    if 'win_rate' in trading_metrics:
                        win_rate = trading_metrics['win_rate']
                        key_metrics.append(f"Win Rate: {win_rate*100:.2f}%")
                    if 'profit_factor' in trading_metrics:
                        profit_factor = trading_metrics['profit_factor']
                        pf_color = Fore.GREEN if profit_factor > 1 else Fore.RED
                        key_metrics.append(f"PF: {pf_color}{profit_factor:.2f}{Style.RESET_ALL}")
                    if 'strategy_return' in trading_metrics:
                        strategy_return = trading_metrics['strategy_return']
                        sr_color = Fore.GREEN if strategy_return > 0 else Fore.RED
                        key_metrics.append(f"Return: {sr_color}{strategy_return*100:.2f}%{Style.RESET_ALL}")

                # Print metrics in a single line
                if key_metrics:
                    print(f"{Fore.YELLOW}METRICS:{Style.RESET_ALL} {' | '.join(key_metrics)}")

            print("=" * width)

        except Exception as e:
            self.print_error(f"Error displaying compact summary: {e}")

    def _get_readme_title(self, file_path):
        """Extract the title from a README file."""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('# '):
                    return first_line[2:]
                return "Documentation file"
        except Exception:
            return "Documentation file"

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
        title = ""
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()
        print(f"{Fore.YELLOW}Please consider donating to support the development of this tool:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}I'll probably just buy a vape or something{Style.RESET_ALL}")
        print()
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


def display_splash_screen():
    """Display a cool ASCII splash screen."""
    # Clear the screen first
    os.system('cls' if os.name == 'nt' else 'clear')

    # Define the ASCII art for the splash screen
    splash = f"""
{Fore.CYAN}    ██████╗ ██╗      █████╗  ██████╗██╗  ██╗███████╗{Style.RESET_ALL}
{Fore.CYAN}    ╚════██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██╔════╝{Style.RESET_ALL}
{Fore.CYAN}     █████╔╝██║     ███████║██║     █████╔╝ ███████╗{Style.RESET_ALL}
{Fore.CYAN}     ╚═══██╗██║     ██╔══██║██║     ██╔═██╗ ╚════██║{Style.RESET_ALL}
{Fore.CYAN}    ██████╔╝███████╗██║  ██║╚██████╗██║  ██╗███████║{Style.RESET_ALL}
{Fore.CYAN}    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝{Style.RESET_ALL}

{Fore.GREEN}    ███████╗ ██████╗ █████╗ ███╗   ██╗███╗   ██╗███████╗██████╗ {Style.RESET_ALL}
{Fore.GREEN}    ██╔════╝██╔════╝██╔══██╗████╗  ██║████╗  ██║██╔════╝██╔══██╗{Style.RESET_ALL}
{Fore.GREEN}    ███████╗██║     ███████║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝{Style.RESET_ALL}
{Fore.GREEN}    ╚════██║██║     ██╔══██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗{Style.RESET_ALL}
{Fore.GREEN}    ███████║╚██████╗██║  ██║██║ ╚████║██║ ╚████║███████╗██║  ██║{Style.RESET_ALL}
{Fore.GREEN}    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝{Style.RESET_ALL}

{Fore.YELLOW}                 Cryptocurrency Technical Analysis Tool{Style.RESET_ALL}
{Fore.BLUE}                      Version 1.0.0 - 2025{Style.RESET_ALL}

"""

    # Print the splash screen
    print(splash)

    # Add a loading animation
    loading_text = "Initializing system components"
    print(f"{Fore.CYAN}{loading_text}...{Style.RESET_ALL}")

    # Simple loading animation
    for _ in range(10):
        time.sleep(0.2)
        print(f"{Fore.CYAN}.{Style.RESET_ALL}", end='', flush=True)
    print("\n")

    # Display system info
    print(f"{Fore.WHITE}System ready. Press Enter to continue...{Style.RESET_ALL}")
    input()

def main():
    """Main function to run the terminal UI."""
    # Display the splash screen
    display_splash_screen()

    # Initialize and run the UI
    ui = TerminalUI()
    ui.run()


if __name__ == "__main__":
    main()
