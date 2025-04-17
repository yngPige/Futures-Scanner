"""
Terminal UI Module for Crypto Futures Scanner

This module provides a terminal-based user interface for the Crypto Futures Scanner application.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from colorama import init, Fore, Back, Style
from src.ui.terminal_output import TerminalOutputGenerator
from src.config import get_pybloat_path, pybloat_file_exists, PYBLOAT_DIR
from src.utils.helpers import (save_settings, load_settings, load_analysis_from_cache,
                             save_analysis_to_cache, get_previous_analyses, load_previous_analysis)

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
            'save': True,
            'tune': False,
            'use_gpu': True   # GPU Acceleration enabled by default
        }

        # Load saved settings if available
        self.load_saved_settings()

        # Track completed functions
        self.completed_functions = {
            'fetch_data': False,
            'train_model': False,
            'make_predictions': False,
            'backtest_strategy': False
        }

        # Create terminal output generator with dark theme
        self.output_generator = TerminalOutputGenerator(theme='dark')

        # LLM functionality removed

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

        self.available_exchanges = [
            'CCXT:ALL', 'kraken', 'kucoin', 'huobi'
        ]

        # Load available models
        self.refresh_available_models()

    # LLM functionality removed

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

    def show_please_wait_animation(self, message="Preparing summary", duration=3):
        """Show a 'Please Wait' animation with a rotating globe.

        Args:
            message (str): Message to display
            duration (int): Duration in seconds
        """
        # ASCII frames for rotating globe animation
        globe_frames = [
            "  o  ",
            "  O  ",
            " (O) ",
            "(O)  ",
            " (O) ",
            "  O  "
        ]

        # ASCII frames for loading bar
        loading_chars = ["▰", "▱"]

        # Clear the screen and print header
        self.clear_screen()
        self.print_header("Blacks Scanner - Analysis Complete")

        # Print the title with the message
        width = 80
        print(f"{Fore.CYAN}{message.center(width)}{Style.RESET_ALL}\n")

        # Calculate animation parameters
        start_time = time.time()
        frame_duration = 0.1  # seconds per frame
        bar_width = 40

        # Run the animation for the specified duration
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / duration)

            # Get the current globe frame
            frame_idx = int((elapsed / frame_duration) % len(globe_frames))
            globe = globe_frames[frame_idx]

            # Create the loading bar
            filled = int(bar_width * progress)
            bar = "".join([loading_chars[0]] * filled + [loading_chars[1]] * (bar_width - filled))

            # Print the animation frame
            print(f"\r{Fore.CYAN}{globe}{Style.RESET_ALL} {Fore.GREEN}{bar}{Style.RESET_ALL} {int(progress * 100)}%", end="")

            # Sleep briefly
            time.sleep(0.1)

        # Clear the animation line
        print("\r" + " " * 100, end="\r")
        print("\n")

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
                "Finalizing analysis..."
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
            ("Save", self.settings['save']),
            ("Tune", self.settings['tune']),
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
                elif desc == "Train Model" and self.completed_functions.get('train_model'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Make Predictions" and self.completed_functions.get('make_predictions'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Backtest Strategy" and self.completed_functions.get('backtest_strategy'):
                    check_mark = f"{Fore.GREEN} ✓{Style.RESET_ALL}"
                elif desc == "Run Analysis" and all(self.completed_functions.values()):
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
            ("1", "Run Analysis"),
            ("2", "Fetch Data"),
            ("3", "Train Model"),
            ("4", "Make Predictions"),
            ("5", "Backtest Strategy"),
            ("p", "Previous Analyses"),
            ("s", "Change Symbol"),
            ("l", "Change Data Limit"),
            ("6", "Settings"),
            ("c", "Clear Data"),
            ("h", "How to Use"),
            ("q", "Quit")
        ]

        self.print_menu_with_settings(menu_items, "Blacks Scanner - Main Menu")

        choice = self.get_input("Enter your choice: ")

        if choice == '1':
            self.run_all_steps()
        elif choice == '2':
            self.fetch_data()
        elif choice == '3':
            self.train_model()
        elif choice == '4':
            self.make_predictions()
        elif choice == '5':
            self.backtest_strategy()
        elif choice.lower() == 'p':
            self.show_previous_analyses()
        elif choice.lower() == 's':
            self.change_symbol()
        elif choice.lower() == 'l':
            self.change_limit()
        elif choice == '6':
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
            ("1", "Change Timeframe"),
            ("2", "Change Data Limit"),
            ("3", "Change Exchange"),
            ("4", "Change Model Type"),
            ("5", "Select Model Path"),
            ("g", "Toggle GPU Acceleration"),
            ("s", "Toggle Save Results"),
            ("t", "Toggle Hyperparameter Tuning"),
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


        self.print_menu_with_settings(menu_items, "Blacks Scanner - Settings")

        choice = self.get_input("Enter your choice: ")

        if choice == '1':
            self.change_timeframe()
        elif choice == '2':
            self.change_limit()
        elif choice == '3':
            self.change_exchange()
        elif choice == '4':
            self.change_model_type()
        elif choice == '5':
            self.select_model_path()


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
            model_filename = os.path.basename(model_path)
            # Shorten long filenames
            if len(model_filename) > 25:
                # Extract the symbol and model type from the filename
                parts = model_filename.split('_')
                if len(parts) >= 3:
                    # Format: symbol_modeltype_timestamp.joblib
                    symbol = parts[0]
                    model_type = parts[1]
                    # Get just the date part of the timestamp (first 8 chars)
                    date_part = parts[2][:8] if len(parts[2]) > 8 else parts[2]
                    model_filename = f"{symbol}_{model_type}_{date_part}.joblib"
            print(f"{i}. {Fore.CYAN}{model_filename}{Style.RESET_ALL}")

        # Display current model path if set
        if self.settings['model_path']:
            current_model = os.path.basename(self.settings['model_path'])
            # Shorten if needed
            if len(current_model) > 25:
                parts = current_model.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    model_type = parts[1]
                    date_part = parts[2][:8] if len(parts[2]) > 8 else parts[2]
                    current_model = f"{symbol}_{model_type}_{date_part}.joblib"
            print(f"\nCurrent model: {Fore.GREEN}{current_model}{Style.RESET_ALL}")
        else:
            print(f"\nNo model currently selected.")

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

    # Theme change functionality removed

    # LLM functionality removed

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
        self.print_header("Blacks Scanner - Download")

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

    # LLM functionality removed

    # TA-Lib installation function removed

    # LLM functionality removed



    # LLM functionality removed

    def build_command_args(self):
        """Build command line arguments from settings."""
        args = argparse.Namespace()

        # Copy settings to args
        for key, value in self.settings.items():
            setattr(args, key, value)

        # LLM functionality removed

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
            from main import fetch_data

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

                # Check if we have cached analysis results
                cached_analysis = load_analysis_from_cache(args.symbol, args.exchange, args.timeframe)

                if cached_analysis is not None:
                    # Use cached analysis
                    df_analyzed = cached_analysis
                    self.print_info(f"Using cached analysis for {args.symbol} on {args.exchange} ({args.timeframe})")

                    # Add a log entry about using cached data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.collected_logs.append((timestamp, f"✓ Using cached analysis data"))
                else:
                    # Analyze the data using the technical analyzer directly
                    from src.analysis.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    df_analyzed = analyzer.analyze(df)

                    # Cache the analysis results for future use
                    if df_analyzed is not None and not df_analyzed.empty:
                        save_analysis_to_cache(df_analyzed, args.symbol, args.exchange, args.timeframe)

                        # Add a log entry about caching the data
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.collected_logs.append((timestamp, f"✓ Saved analysis to cache for faster future access"))

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
                    # Analysis completed

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
                        # middle band is used for reference but not directly displayed
                        # middle = latest['BBM_20_2.0']
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
            from main import fetch_data, train_model

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

                # Check if we have cached analysis results
                cached_analysis = load_analysis_from_cache(args.symbol, args.exchange, args.timeframe)

                if cached_analysis is not None:
                    # Use cached analysis
                    df_analyzed = cached_analysis
                    self.print_info(f"Using cached analysis for {args.symbol} on {args.exchange} ({args.timeframe})")

                    # Add a log entry about using cached data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.collected_logs.append((timestamp, f"✓ Using cached analysis data"))
                else:
                    # Analyze the data using the technical analyzer directly
                    from src.analysis.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    df_analyzed = analyzer.analyze(df)

                    # Cache the analysis results for future use
                    if df_analyzed is not None and not df_analyzed.empty:
                        save_analysis_to_cache(df_analyzed, args.symbol, args.exchange, args.timeframe)

                        # Add a log entry about caching the data
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.collected_logs.append((timestamp, f"✓ Saved analysis to cache for faster future access"))

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
                            # Get a shorter version of the model path filename
                            model_filename = os.path.basename(model_path)
                            if model_filename and len(model_filename) > 25:
                                # Extract the symbol and model type from the filename
                                parts = model_filename.split('_')
                                if len(parts) >= 3:
                                    # Format: symbol_modeltype_timestamp.joblib
                                    symbol = parts[0]
                                    model_type = parts[1]
                                    # Get just the date part of the timestamp (first 8 chars)
                                    date_part = parts[2][:8] if len(parts[2]) > 8 else parts[2]
                                    model_filename = f"{symbol}_{model_type}_{date_part}.joblib"

                            print(f"Model: {Fore.CYAN}{model_filename}{Style.RESET_ALL}")
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
            from main import fetch_data, predict

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

                # Check if we have cached analysis results
                cached_analysis = load_analysis_from_cache(args.symbol, args.exchange, args.timeframe)

                if cached_analysis is not None:
                    # Use cached analysis
                    df_analyzed = cached_analysis
                    self.print_info(f"Using cached analysis for {args.symbol} on {args.exchange} ({args.timeframe})")

                    # Add a log entry about using cached data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.collected_logs.append((timestamp, f"✓ Using cached analysis data"))
                else:
                    # Analyze the data using the technical analyzer directly
                    from src.analysis.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    df_analyzed = analyzer.analyze(df)

                    # Cache the analysis results for future use
                    if df_analyzed is not None and not df_analyzed.empty:
                        save_analysis_to_cache(df_analyzed, args.symbol, args.exchange, args.timeframe)

                        # Add a log entry about caching the data
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.collected_logs.append((timestamp, f"✓ Saved analysis to cache for faster future access"))

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

                        # Skip the immediate prediction summary display
                        # We'll let the display_in_current_terminal method handle the full display

                        # Display predictions in current terminal
                        self.print_info("\nPreparing prediction results...")

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
            from main import fetch_data, predict, backtest

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

                # Check if we have cached analysis results
                cached_analysis = load_analysis_from_cache(args.symbol, args.exchange, args.timeframe)

                if cached_analysis is not None:
                    # Use cached analysis
                    df_analyzed = cached_analysis
                    self.print_info(f"Using cached analysis for {args.symbol} on {args.exchange} ({args.timeframe})")

                    # Add a log entry about using cached data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.collected_logs.append((timestamp, f"✓ Using cached analysis data"))
                else:
                    # Analyze the data using the technical analyzer directly
                    from src.analysis.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    df_analyzed = analyzer.analyze(df)

                    # Cache the analysis results for future use
                    if df_analyzed is not None and not df_analyzed.empty:
                        save_analysis_to_cache(df_analyzed, args.symbol, args.exchange, args.timeframe)

                        # Add a log entry about caching the data
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        self.collected_logs.append((timestamp, f"✓ Saved analysis to cache for faster future access"))

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

    # LLM functionality removed

    def run_all_steps(self):  # Function name kept for compatibility
        """Run analysis based on current settings and display results at the end."""
        self.print_header("Running Analysis")

        try:
            # Import main from PyBloat directory
            self._import_from_pybloat()
            from main import fetch_data, train_model, predict, backtest

            args = self.build_command_args()

            # Clear any previous logs
            self.collected_logs = []

            # Create a dictionary to store all results for display at the end
            all_results = {}
            all_results['start_time'] = datetime.now()

            # Fetch data
            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")

            # Show loading animation while fetching data
            self.show_loading_animation("Retrieving market data", duration=2, compact_completion=True)

            df = fetch_data(args)
            if df is None:
                self.print_error("Failed to fetch data.")
                self.wait_for_key()
                return

            # Mark fetch_data as completed
            self.completed_functions['fetch_data'] = True

            # Store data information for later display
            all_results['data'] = {
                'rows': len(df),
                'time_range': (df.index.min(), df.index.max()),
                'exchange': df.attrs.get('exchange', self.settings['exchange'])
            }

            # Analyze data
            self.print_info("Performing technical analysis...")

            # Show enhanced loading animation with detailed technical analysis logs
            analysis_logs = [
                "Initializing technical analysis engine...",
                "Standardizing price data columns...",
                "Computing trend indicators...",
                "Calculating momentum oscillators...",
                "Generating MACD signals...",
                "Computing Bollinger Bands...",
                "Calculating volatility metrics...",
                "Identifying support/resistance levels...",
                "Detecting price patterns...",
                "Computing volume indicators...",
                "Calculating Ichimoku Cloud...",
                "Generating Fibonacci levels...",
                "Identifying divergence patterns...",
                "Calculating pivot points...",
                "Generating trading signals..."
            ]
            self.show_loading_animation("Calculating technical indicators", duration=3, log_messages=analysis_logs, compact_completion=True)

            # Check if we have cached analysis results
            cached_analysis = load_analysis_from_cache(args.symbol, args.exchange, args.timeframe)

            if cached_analysis is not None:
                # Use cached analysis
                df_analyzed = cached_analysis
                self.print_info(f"Using cached analysis for {args.symbol} on {args.exchange} ({args.timeframe})")

                # Add a log entry about using cached data
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.collected_logs.append((timestamp, f"✓ Using cached analysis data"))
            else:
                # Analyze the data using the technical analyzer directly
                from src.analysis.technical_analysis import TechnicalAnalyzer
                analyzer = TechnicalAnalyzer()
                df_analyzed = analyzer.analyze(df)

                # Cache the analysis results for future use
                if df_analyzed is not None and not df_analyzed.empty:
                    save_analysis_to_cache(df_analyzed, args.symbol, args.exchange, args.timeframe)

                    # Add a log entry about caching the data
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.collected_logs.append((timestamp, f"✓ Saved analysis to cache for faster future access"))

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

                # Store indicator information for later display
                all_results['indicators'] = {
                    'count': len(df_analyzed.columns),
                    'groups': {group: len(indicators) for group, indicators in indicator_groups.items() if indicators}
                }

                # Analysis step is integrated into the process
            if df_analyzed is None or df_analyzed.empty:
                self.print_error("Analysis resulted in empty DataFrame or failed.")
                self.wait_for_key()
                return

            # Train model with enhanced loading animation
            self.print_info(f"Training {args.model_type.replace('_', ' ').title()} model...")

            # Define custom logs for model training
            train_logs = [
                "Preparing training dataset...",
                "Splitting data into train/test sets...",
                "Scaling features...",
                "Initializing model architecture...",
                "Training model...",
                "Optimizing parameters...",
                "Evaluating performance...",
                "Finalizing model..."
            ]
            self.show_loading_animation("Training machine learning model", duration=4, log_messages=train_logs, compact_completion=True)

            model, model_path = train_model(df_analyzed, args)
            if model is None:
                self.print_error("Failed to train model.")
                self.wait_for_key()
                return

            # Store model information for later display
            feature_count = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown'

            # Get a shorter version of the model path filename
            model_filename = os.path.basename(model_path) if model_path else None
            if model_filename and len(model_filename) > 25:
                # Extract the symbol and model type from the filename
                parts = model_filename.split('_')
                if len(parts) >= 3:
                    # Format: symbol_modeltype_timestamp.joblib
                    symbol = parts[0]
                    model_type = parts[1]
                    # Get just the date part of the timestamp (first 8 chars)
                    date_part = parts[2][:8] if len(parts[2]) > 8 else parts[2]
                    model_filename = f"{symbol}_{model_type}_{date_part}.joblib"

            all_results['model'] = {
                'type': args.model_type.replace('_', ' ').title(),
                'features': feature_count,
                'tuning': args.tune,
                'path': model_filename
            }

            # Update settings
            if model_path:
                self.settings['model_path'] = model_path
                self.refresh_available_models()

            # Mark train_model as completed
            self.completed_functions['train_model'] = True

            # Make predictions
            self.print_info("Making predictions...")

            # Show loading animation while making predictions
            self.show_loading_animation("Running prediction model", duration=2, compact_completion=True)

            df_predictions = predict(df_analyzed, model_path, args)
            if df_predictions is None:
                self.print_error("Failed to make predictions.")
                self.wait_for_key()
                return

            # Store prediction information for later display
            if 'prediction' in df_predictions.columns:
                # Count predictions
                pred_counts = df_predictions['prediction'].value_counts()
                total_preds = len(df_predictions)

                # Calculate percentages
                bullish_count = pred_counts.get(1, 0)
                bearish_count = pred_counts.get(0, 0)
                bullish_pct = (bullish_count / total_preds) * 100 if total_preds > 0 else 0
                bearish_pct = (bearish_count / total_preds) * 100 if total_preds > 0 else 0

                all_results['predictions'] = {
                    'bullish_count': bullish_count,
                    'bearish_count': bearish_count,
                    'bullish_pct': bullish_pct,
                    'bearish_pct': bearish_pct
                }

                # Mark make_predictions as completed
                self.completed_functions['make_predictions'] = True

            # Backtest
            self.print_info("Backtesting predictions...")

            # Show loading animation while backtesting
            self.show_loading_animation("Simulating trading strategy", duration=3, compact_completion=True)

            performance_metrics, trading_metrics = backtest(df_predictions, args)

            # Store metrics for later display
            all_results['performance_metrics'] = performance_metrics
            all_results['trading_metrics'] = trading_metrics

            # Mark backtest_strategy as completed
            self.completed_functions['backtest_strategy'] = True

            # Perform LLM analysis
            self.print_info("Performing AI market analysis...")

            # Show loading animation while performing LLM analysis
            llm_logs = [
                "Initializing AI analysis engine...",
                "Processing market data...",
                "Analyzing technical indicators...",
                "Evaluating market sentiment...",
                "Identifying key price levels...",
                "Generating trading insights...",
                "Finalizing market analysis..."
            ]
            self.show_loading_animation("Running AI market analysis", duration=3, log_messages=llm_logs, compact_completion=True)

            # Import the LLM analyzer
            from src.analysis.llm_analysis import LLMAnalyzer
            llm_analyzer = LLMAnalyzer()

            # Perform the analysis
            llm_results = llm_analyzer.analyze_market(df_analyzed, args.symbol, args.timeframe)

            # Check if we're using fallback analysis
            if llm_results.get('is_fallback', False):
                self.print_warning("Using fallback analysis - Ollama not available or model not loaded.")
                self.print_info("To use the full AI analysis, install Ollama and run: 'ollama pull llama3'")
            elif 'error' in llm_results:
                self.print_warning(f"LLM analysis encountered an issue: {llm_results['error']}")

            # Store the results
            all_results['llm_analysis'] = llm_results

            # All functions should be marked as completed at their respective steps

            # Clear the screen and show a please wait animation before displaying results
            self.show_please_wait_animation("Preparing Analysis Summary", duration=2)

            # Clear previous logs except completions
            if hasattr(self, 'collected_logs'):
                self.collected_logs = [log for log in self.collected_logs if any(keyword in log[1].lower() for keyword in ['completed', 'success', 'finished', '✓'])]

            # Display data information
            print(f"Symbol: {args.symbol} | Timeframe: {args.timeframe} | Exchange: {all_results['data']['exchange']}")
            print(f"Data range: {all_results['data']['time_range'][0]} to {all_results['data']['time_range'][1]}")

            # Display model information
            if 'model' in all_results:
                model_info = all_results['model']
                print(f"\nModel: {model_info['type']} | Features: {model_info['features']} | Tuning: {'Enabled' if model_info['tuning'] else 'Disabled'}")
                if model_info['path']:
                    print(f"Model: {Fore.CYAN}{model_info['path']}{Style.RESET_ALL}")

            # Display prediction summary
            if 'predictions' in all_results:
                pred_info = all_results['predictions']
                print("\nPrediction Summary:")
                print(f"Bullish signals: {pred_info['bullish_count']} ({pred_info['bullish_pct']:.1f}%)")
                print(f"Bearish signals: {pred_info['bearish_count']} ({pred_info['bearish_pct']:.1f}%)")

            # Display a compact summary with all metrics
            self._display_compact_summary(df_predictions, args.symbol, args.timeframe,
                                        all_results['performance_metrics'], all_results['trading_metrics'])

            # Display LLM analysis
            if 'llm_analysis' in all_results and 'analysis' in all_results['llm_analysis']:
                self._display_llm_analysis(all_results['llm_analysis'], args.symbol, args.timeframe)



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
                visualize(df_analyzed, args)
            elif chart_choice == 'b':
                # Use browser chart
                args.terminal_chart = False
                args.interactive = True
                self.print_info("Displaying chart in browser...")
                # Import main from PyBloat directory
                self._import_from_pybloat()
                from main import visualize
                visualize(df_analyzed, args)

            self.print_success("\nAnalysis completed successfully!")

        except Exception as e:
            self.print_error(f"Error running all steps: {e}")
            import traceback
            self.print_error(traceback.format_exc())

        self.wait_for_key()

    def display_collected_logs(self, title="Process Logs", operation_start_time=None):
        """Display collected logs in a concise format, showing only completion messages.

        Args:
            title (str): Title for the log display
            operation_start_time (datetime, optional): If provided, only show logs after this time
        """
        if not hasattr(self, 'collected_logs') or not self.collected_logs:
            return

        # Filter logs to only show completion messages
        completion_keywords = ['completed', 'success', 'finished', '✓']
        completion_logs = []

        for timestamp_str, message in self.collected_logs:
            # Check if this is a completion message
            if any(keyword in message.lower() for keyword in completion_keywords):
                # Filter by start time if provided
                if operation_start_time:
                    try:
                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
                        # Set the date to today (since log timestamps don't include date)
                        today = datetime.now().date()
                        timestamp = datetime.combine(today, timestamp.time())

                        # Only include logs after the start time
                        if timestamp >= operation_start_time:
                            completion_logs.append((timestamp_str, message))
                    except ValueError:
                        # If timestamp parsing fails, include the log anyway
                        completion_logs.append((timestamp_str, message))
                else:
                    completion_logs.append((timestamp_str, message))

        # If we have completion logs, display them
        if completion_logs:
            # Print header
            width = 80
            print("\n" + "=" * width)
            print(f"{Fore.CYAN}{title:^{width}}{Style.RESET_ALL}")
            print("=" * width)

            # Print completion logs
            for timestamp, message in sorted(completion_logs, key=lambda x: x[0]):
                print(f"  {Fore.WHITE}[{timestamp}]{Style.RESET_ALL} {Fore.GREEN}{message}{Style.RESET_ALL}")

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

    def _display_llm_analysis(self, llm_results, symbol, timeframe):
        """
        Display LLM analysis results in a formatted box.

        Args:
            llm_results (dict): Results from the LLM analysis
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the analysis
        """
        try:
            # Create a box for the LLM analysis
            width = 80
            print("\n" + "=" * width)
            print(f"{Fore.CYAN}AI MARKET ANALYSIS - {symbol} ({timeframe}){Style.RESET_ALL}".center(width))
            print("=" * width)

            # Check if there was an error or using fallback
            if llm_results.get('is_fallback', False):
                print(f"{Fore.YELLOW}Note: Using algorithmic analysis. Full AI analysis requires Ollama.{Style.RESET_ALL}")
                print(f"{Fore.CYAN}To enable AI analysis: Install Ollama and run 'ollama pull llama3'{Style.RESET_ALL}\n")
            elif 'error' in llm_results:
                print(f"{Fore.YELLOW}Note: {llm_results['error']}{Style.RESET_ALL}\n")

            # Display sentiment with color
            sentiment = llm_results.get('sentiment', 'neutral').lower()
            sentiment_color = Fore.WHITE
            if sentiment == 'bullish':
                sentiment_color = Fore.GREEN
            elif sentiment == 'bearish':
                sentiment_color = Fore.RED
            elif sentiment == 'neutral':
                sentiment_color = Fore.YELLOW

            print(f"{Fore.WHITE}Market Sentiment: {sentiment_color}{sentiment.capitalize()}{Style.RESET_ALL}")

            # Display risk level with color
            risk = llm_results.get('risk', 'medium').lower()
            risk_color = Fore.WHITE
            if risk == 'low':
                risk_color = Fore.GREEN
            elif risk == 'medium':
                risk_color = Fore.YELLOW
            elif risk == 'high':
                risk_color = Fore.RED

            print(f"{Fore.WHITE}Risk Level: {risk_color}{risk.capitalize()}{Style.RESET_ALL}")

            # Display timestamp
            if 'timestamp' in llm_results:
                print(f"{Fore.WHITE}Analysis Time: {Fore.CYAN}{llm_results['timestamp']}{Style.RESET_ALL}")

            print("\n" + "-" * width)

            # Format and display the analysis text
            analysis_text = llm_results.get('analysis', 'No analysis available.')

            # Split the text into paragraphs
            paragraphs = analysis_text.split('\n\n')

            for paragraph in paragraphs:
                # Check if this is a numbered list item
                if paragraph.strip() and paragraph.strip()[0].isdigit() and ". " in paragraph:
                    # Print each line of the paragraph
                    for line in paragraph.split('\n'):
                        if line.strip():
                            # Highlight numbers in lists
                            if line.strip()[0].isdigit() and ". " in line:
                                parts = line.split(". ", 1)
                                print(f"{Fore.CYAN}{parts[0]}.{Style.RESET_ALL} {parts[1]}")
                            else:
                                print(line)
                else:
                    # Print the paragraph with word wrapping
                    self._print_wrapped_text(paragraph, width - 4)

                # Add spacing between paragraphs
                print()

            print("=" * width)

        except Exception as e:
            print(f"{Fore.RED}Error displaying LLM analysis: {e}{Style.RESET_ALL}")

    def _print_wrapped_text(self, text, width):
        """
        Print text with word wrapping.

        Args:
            text (str): Text to print
            width (int): Maximum width for each line
        """
        # Split the text into words
        words = text.split()
        if not words:
            print()
            return

        # Build lines word by word
        current_line = words[0]
        for word in words[1:]:
            # If adding the next word would exceed the width, print the current line and start a new one
            if len(current_line) + len(word) + 1 > width:
                print(current_line)
                current_line = word
            else:
                current_line += " " + word

        # Print the last line
        if current_line:
            print(current_line)

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

        if 'models' in directories_to_clear:
            self.completed_functions['train_model'] = False

        if 'results' in directories_to_clear:
            self.completed_functions['make_predictions'] = False
            self.completed_functions['backtest_strategy'] = False

        # Update available models list if models directory was cleared
        if 'models' in directories_to_clear:
            self.refresh_available_models()

        self.wait_for_key()

    def set_terminal_size(self, cols=80, lines=30):
        """Set the terminal window size.

        Args:
            cols (int): Number of columns (width)
            lines (int): Number of lines (height)
        """
        try:
            # On Windows, use mode command to set terminal size
            if os.name == 'nt':
                os.system(f'mode con: cols={cols} lines={lines}')
            # Unix systems would use stty, but we're focusing on Windows for now
        except Exception as e:
            logger.error(f"Error setting terminal size: {e}")

    def show_previous_analyses(self):
        """Show previous analyses."""
        self.print_header("Previous Analyses")

        # Get previous analyses
        analyses = get_previous_analyses()

        if not analyses:
            self.print_info("No previous analyses found.")
            self.wait_for_key()
            return

        # Group analyses by symbol
        symbols = sorted(list(set([info['symbol'] for _, files in analyses.items() for info in files])))

        # Display available symbols
        self.print_info("Available symbols with previous analyses:")
        for i, symbol in enumerate(symbols, 1):
            print(f"{Fore.GREEN}{i}{Style.RESET_ALL}: {symbol}")

        print(f"{Fore.GREEN}b{Style.RESET_ALL}: Back to main menu")

        # Get user choice
        choice = self.get_input("\nSelect a symbol: ")

        if choice.lower() == 'b':
            return

        try:
            symbol_index = int(choice) - 1
            if symbol_index < 0 or symbol_index >= len(symbols):
                self.print_error("Invalid choice.")
                time.sleep(1)
                return

            selected_symbol = symbols[symbol_index]
            self.show_symbol_analyses(selected_symbol, analyses)

        except ValueError:
            self.print_error("Invalid choice.")
            time.sleep(1)

    def show_symbol_analyses(self, symbol, analyses=None):
        """Show analyses for a specific symbol."""
        if analyses is None:
            analyses = get_previous_analyses()

        self.print_header(f"Previous Analyses for {symbol}")

        # Filter analyses for the selected symbol
        symbol_analyses = {}
        for _, files in analyses.items():
            for file_info in files:
                if file_info['symbol'] == symbol:
                    timeframe = file_info['timeframe']
                    if timeframe not in symbol_analyses:
                        symbol_analyses[timeframe] = []
                    symbol_analyses[timeframe].append(file_info)

        if not symbol_analyses:
            self.print_info(f"No analyses found for {symbol}.")
            self.wait_for_key()
            return

        # Display available timeframes
        self.print_info("Available timeframes:")
        timeframes = sorted(symbol_analyses.keys())
        for i, timeframe in enumerate(timeframes, 1):
            print(f"{Fore.GREEN}{i}{Style.RESET_ALL}: {timeframe} ({len(symbol_analyses[timeframe])} analyses)")

        print(f"{Fore.GREEN}b{Style.RESET_ALL}: Back to symbol selection")

        # Get user choice
        choice = self.get_input("\nSelect a timeframe: ")

        if choice.lower() == 'b':
            self.show_previous_analyses()
            return

        try:
            timeframe_index = int(choice) - 1
            if timeframe_index < 0 or timeframe_index >= len(timeframes):
                self.print_error("Invalid choice.")
                time.sleep(1)
                return

            selected_timeframe = timeframes[timeframe_index]
            self.show_timeframe_analyses(symbol, selected_timeframe, symbol_analyses[selected_timeframe])

        except ValueError:
            self.print_error("Invalid choice.")
            time.sleep(1)

    def show_timeframe_analyses(self, symbol, timeframe, analyses):
        """Show analyses for a specific symbol and timeframe."""
        self.print_header(f"Previous Analyses for {symbol} ({timeframe})")

        # Group analyses by type
        analyses_by_type = {}
        for analysis in analyses:
            file_type = analysis['type']
            if file_type not in analyses_by_type:
                analyses_by_type[file_type] = []
            analyses_by_type[file_type].append(analysis)

        # Display available analysis types
        self.print_info("Available analysis types:")
        types = sorted(analyses_by_type.keys())
        for i, analysis_type in enumerate(types, 1):
            print(f"{Fore.GREEN}{i}{Style.RESET_ALL}: {analysis_type} ({len(analyses_by_type[analysis_type])} files)")

        print(f"{Fore.GREEN}b{Style.RESET_ALL}: Back to timeframe selection")

        # Get user choice
        choice = self.get_input("\nSelect an analysis type: ")

        if choice.lower() == 'b':
            self.show_symbol_analyses(symbol)
            return

        try:
            type_index = int(choice) - 1
            if type_index < 0 or type_index >= len(types):
                self.print_error("Invalid choice.")
                time.sleep(1)
                return

            selected_type = types[type_index]
            self.show_analysis_files(symbol, timeframe, selected_type, analyses_by_type[selected_type])

        except ValueError:
            self.print_error("Invalid choice.")
            time.sleep(1)

    def show_analysis_files(self, symbol, timeframe, analysis_type, analyses):
        """Show analysis files for a specific symbol, timeframe, and type."""
        self.print_header(f"{analysis_type.title()} Analyses for {symbol} ({timeframe})")

        # Sort analyses by datetime (most recent first)
        analyses.sort(key=lambda x: x['datetime'], reverse=True)

        # Display available analysis files
        self.print_info("Available analysis files:")
        for i, analysis in enumerate(analyses, 1):
            # Format the datetime
            date_str = analysis['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            print(f"{Fore.GREEN}{i}{Style.RESET_ALL}: {date_str} - {analysis['filename']}")

        print(f"{Fore.GREEN}b{Style.RESET_ALL}: Back to analysis type selection")

        # Get user choice
        choice = self.get_input("\nSelect an analysis file: ")

        if choice.lower() == 'b':
            self.show_timeframe_analyses(symbol, timeframe, analyses)
            return

        try:
            file_index = int(choice) - 1
            if file_index < 0 or file_index >= len(analyses):
                self.print_error("Invalid choice.")
                time.sleep(1)
                return

            selected_file = analyses[file_index]
            self.display_analysis_file(selected_file)

        except ValueError:
            self.print_error("Invalid choice.")
            time.sleep(1)

    def display_analysis_file(self, file_info):
        """Display the contents of an analysis file."""
        self.print_header(f"Analysis File: {file_info['filename']}")

        # Load the file
        data, file_type = load_previous_analysis(file_info['path'])

        if data is None:
            self.print_error(f"Failed to load file: {file_info['path']}")
            self.wait_for_key()
            return

        # Display the file contents based on type
        if file_type == 'dataframe':
            # Display DataFrame summary
            self.print_info(f"DataFrame with {len(data)} rows and {len(data.columns)} columns")
            print("\nColumns:")
            for col in data.columns:
                print(f"  {col}")

            print("\nSample data:")
            print(data.head().to_string())

            # Display key indicators if available
            latest = data.iloc[-1] if not data.empty else None
            if latest is not None:
                print("\nKey indicators:")

                # Show RSI if available
                if 'rsi_14' in data.columns:
                    rsi = latest['rsi_14']
                    rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                    rsi_color = Fore.RED if rsi < 30 else Fore.GREEN if rsi > 70 else Fore.WHITE
                    print(f"RSI (14): {rsi_color}{rsi:.2f} - {rsi_status}{Style.RESET_ALL}")

                # Show MACD if available
                if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
                    macd = latest['MACD_12_26_9']
                    signal = latest['MACDs_12_26_9']
                    macd_hist = macd - signal
                    macd_status = "Bullish" if macd > signal else "Bearish"
                    macd_color = Fore.GREEN if macd > signal else Fore.RED
                    print(f"MACD: {macd_color}{macd:.4f} - {macd_status}{Style.RESET_ALL} (H: {macd_hist:.4f})")

                # Show Moving Averages if available
                if 'sma_50' in data.columns and 'sma_200' in data.columns:
                    sma_50 = latest['sma_50']
                    sma_200 = latest['sma_200']
                    ma_status = "Golden Cross" if sma_50 > sma_200 else "Death Cross"
                    ma_color = Fore.GREEN if sma_50 > sma_200 else Fore.RED
                    print(f"MA Cross: {ma_color}{ma_status}{Style.RESET_ALL} ({sma_50:.2f}/{sma_200:.2f})")

                # Show prediction if available
                if 'prediction' in data.columns:
                    pred_value = latest['prediction']
                    pred_prob = latest.get('prediction_probability', 0.5)
                    pred_text = "Bullish" if pred_value == 1 else "Bearish"
                    pred_color = Fore.GREEN if pred_value == 1 else Fore.RED
                    print(f"Prediction: {pred_color}{pred_text}{Style.RESET_ALL} ({pred_prob:.2f})")

        elif file_type == 'json':
            # Display JSON data
            if isinstance(data, dict):
                # Format the JSON data for display
                for key, value in data.items():
                    if key == 'performance' or key == 'trading':
                        print(f"\n{Fore.YELLOW}{key.title()} Metrics:{Style.RESET_ALL}")
                        for metric_key, metric_value in value.items():
                            print(f"  {metric_key}: {metric_value}")
                    elif key == 'analysis':
                        print(f"\n{Fore.YELLOW}Analysis:{Style.RESET_ALL}")
                        print(value)
                    elif key == 'sentiment':
                        sentiment_color = Fore.GREEN if value == 'bullish' else Fore.RED if value == 'bearish' else Fore.YELLOW
                        print(f"\n{Fore.YELLOW}Sentiment:{Style.RESET_ALL} {sentiment_color}{value}{Style.RESET_ALL}")
                    elif key == 'risk':
                        risk_color = Fore.GREEN if value == 'low' else Fore.RED if value == 'high' else Fore.YELLOW
                        print(f"\n{Fore.YELLOW}Risk:{Style.RESET_ALL} {risk_color}{value}{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.YELLOW}{key.title()}:{Style.RESET_ALL} {value}")
            else:
                # Just print the JSON data
                print(json.dumps(data, indent=2))

        elif file_type == 'text':
            # Display text data
            print(data)

        elif file_type == 'pickle':
            # Display pickle data summary
            if isinstance(data, dict):
                print(f"Dictionary with {len(data)} keys:")
                for key in data.keys():
                    print(f"  {key}")
            elif isinstance(data, pd.DataFrame):
                print(f"DataFrame with {len(data)} rows and {len(data.columns)} columns")
                print("\nColumns:")
                for col in data.columns:
                    print(f"  {col}")
                print("\nSample data:")
                print(data.head().to_string())
            else:
                print(f"Data type: {type(data)}")
                print(str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data))

        # Ask if the user wants to use this analysis for the current session
        print("\n" + "=" * 80)
        choice = self.get_input("\nUse this analysis for the current session? (y/n): ")

        if choice.lower() == 'y':
            # Update settings with the symbol and timeframe from the file
            self.settings['symbol'] = file_info['symbol']
            self.settings['timeframe'] = file_info['timeframe']
            self.settings['exchange'] = self.settings.get('exchange', 'kraken')  # Use current exchange or default to kraken

            # Save settings
            self.save_current_settings()

            self.print_success(f"Updated current session to use {file_info['symbol']} on {file_info['timeframe']} timeframe.")

        self.wait_for_key()

    def load_saved_settings(self):
        """Load saved settings from file."""
        try:
            saved_settings = load_settings()
            if saved_settings:
                # Update settings with saved values, keeping defaults for any missing keys
                for key, value in saved_settings.items():
                    if key in self.settings:
                        self.settings[key] = value
                self.print_info(f"Loaded saved settings for {self.settings['symbol']} on {self.settings['exchange']}")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")

    def save_current_settings(self):
        """Save current settings to file."""
        try:
            # Save settings
            save_settings(self.settings)
            logger.info(f"Saved settings for {self.settings['symbol']} on {self.settings['exchange']}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def show_exit_screen(self):
        """Show a simple exit screen."""
        # Save current settings before exiting
        self.save_current_settings()

        # Set terminal size
        self.set_terminal_size(80, 30)

        self.clear_screen()
        width = 80
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        title = "Thanks for using Blacks Scanner"
        print(Fore.CYAN + f"{title:^{width}}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print()

        # Simple goodbye message
        print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
        print()

        # Display a countdown for 5 seconds
        for i in range(5, 0, -1):
            print(f"Exiting in {i} seconds...\r", end="")
            time.sleep(1)

        print("\nPress Enter to exit...")
        input()

    def run(self):
        """Run the terminal UI."""
        while self.running:
            self.current_menu()


# Terminal window size is now handled by the TerminalUI.set_terminal_size method

def display_splash_screen():
    """Display a cool ASCII splash screen."""
    # Set the window size to match the screenshot dimensions
    # Use os.system directly since this is outside the class
    try:
        # On Windows, use mode command to set terminal size
        if os.name == 'nt':
            os.system('mode con: cols=100 lines=30')
        # Unix systems would use stty, but we're focusing on Windows for now
    except Exception as e:
        logger.error(f"Error setting terminal size: {e}")

    # Clear the screen first
    os.system('cls' if os.name == 'nt' else 'clear')

    # Define the ASCII art for the splash screen using standard ASCII characters
    splash = f"""
{Fore.CYAN}   ____  _               _        {Style.RESET_ALL}
{Fore.CYAN}  | __ )| | __ _  ___  | | _____ {Style.RESET_ALL}
{Fore.CYAN}  |  _ \| |/ _` |/ __| | |/ / __|{Style.RESET_ALL}
{Fore.CYAN}  | |_) | | (_| | (__  |   <\__ \{Style.RESET_ALL}
{Fore.CYAN}  |____/|_|\__,_|\___| |_|\_\___/{Style.RESET_ALL}

{Fore.GREEN}   ____                                 {Style.RESET_ALL}
{Fore.GREEN}  / ___|  ___ __ _ _ __  _ __   ___ _ __ {Style.RESET_ALL}
{Fore.GREEN}  \___ \ / __/ _` | '_ \| '_ \ / _ \ '__|{Style.RESET_ALL}
{Fore.GREEN}   ___) | (_| (_| | | | | | | |  __/ |   {Style.RESET_ALL}
{Fore.GREEN}  |____/ \___\__,_|_| |_|_| |_|\___|_|   {Style.RESET_ALL}

{Fore.YELLOW}                Cryptocurrency Technical Analysis Tool{Style.RESET_ALL}
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

    # The UI will automatically load saved settings in its __init__ method
    # and save settings when exiting

    ui.run()


if __name__ == "__main__":
    main()
