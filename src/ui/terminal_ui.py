"""
Terminal UI Module for Crypto Futures Scanner

This module provides a terminal-based user interface for the Crypto Futures Scanner application.
"""

import os
import time
import logging
import argparse
from colorama import init, Fore, Style
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
            'exchange': 'binance',
            'model_type': 'random_forest',
            'model_path': None,
            'theme': 'dark',
            'save': True,
            'tune': False
        }

        # Create terminal output generator
        self.output_generator = TerminalOutputGenerator(theme=self.settings['theme'])

        # Available options
        self.available_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT',
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD',
            'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD'
        ]

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

    def print_settings(self):
        """Print the current settings."""
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

    def main_menu(self):
        """Display the main menu."""
        self.print_header("Crypto Futures Scanner - Main Menu")

        self.print_menu_item("1", "Fetch Data")
        self.print_menu_item("2", "Analyze Data")
        self.print_menu_item("3", "Train Model")
        self.print_menu_item("4", "Make Predictions")
        self.print_menu_item("5", "Backtest Strategy")
        self.print_menu_item("6", "Run All Steps")
        self.print_menu_item("7", "Settings")
        self.print_menu_item("q", "Quit")

        self.print_settings()

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
            self.current_menu = self.settings_menu
        elif choice.lower() == 'q':
            self.running = False
        else:
            self.print_error("Invalid choice. Please try again.")
            time.sleep(1)

    def settings_menu(self):
        """Display the settings menu."""
        self.print_header("Crypto Futures Scanner - Settings")

        self.print_menu_item("1", "Change Symbol")
        self.print_menu_item("2", "Change Timeframe")
        self.print_menu_item("3", "Change Data Limit")
        self.print_menu_item("4", "Change Exchange")
        self.print_menu_item("5", "Change Model Type")
        self.print_menu_item("6", "Select Model Path")
        self.print_menu_item("7", "Change Theme")
        self.print_menu_item("9", "Toggle Save Results")
        self.print_menu_item("t", "Toggle Hyperparameter Tuning")
        self.print_menu_item("b", "Back to Main Menu")

        self.print_settings()

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

        elif choice == '9':
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

        print("Available symbols:")
        for i, symbol in enumerate(self.available_symbols, 1):
            print(f"{i}. {symbol}")

        print(f"\nCurrent symbol: {self.settings['symbol']}")

        choice = self.get_input("\nEnter symbol number or custom symbol (or 'b' to go back): ")

        if choice.lower() == 'b':
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(self.available_symbols):
                self.settings['symbol'] = self.available_symbols[index]
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
        exchanges = ['binance', 'coinbase', 'kraken', 'kucoin', 'huobi', 'bybit']
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

    def build_command_args(self):
        """Build command line arguments from settings."""
        args = argparse.Namespace()

        # Copy settings to args
        for key, value in self.settings.items():
            setattr(args, key, value)

        return args

    def fetch_data(self):
        """Fetch data based on current settings."""
        self.print_header("Fetching Data")

        try:
            from main import fetch_data

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is not None:
                self.print_success(f"Successfully fetched {len(df)} rows of data.")

                # Display sample data
                print("\nSample data:")
                print(df.head().to_string())
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

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")
                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_success(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators.")

                    # Display sample data
                    print("\nSample analyzed data:")
                    print(df_analyzed.head().to_string())

                    # Generate and open terminal output
                    self.print_info("Opening analysis in new terminal window...")
                    script_path = self.output_generator.generate_output(
                        df_analyzed, args.symbol, args.timeframe, report_type='analysis'
                    )
                    if script_path:
                        if self.output_generator.open_terminal(script_path):
                            self.print_success("Analysis opened in new terminal window.")
                        else:
                            self.print_error("Failed to open terminal window.")
                    else:
                        self.print_error("Failed to generate analysis output.")
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

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")
                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info(f"Training {args.model_type} model...")
                    model, model_path = train_model(df_analyzed, args)

                    if model is not None:
                        self.print_success("Successfully trained model.")
                        if model_path:
                            self.print_success(f"Model saved to {model_path}")
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

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")
                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info("Making predictions...")
                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_success("Successfully made predictions.")

                        # Display sample data
                        print("\nSample predictions:")
                        print(df_predictions[['close', 'prediction', 'prediction_probability']].tail().to_string())

                        # Generate and open terminal output
                        self.print_info("Opening predictions in new terminal window...")
                        script_path = self.output_generator.generate_output(
                            df_predictions, args.symbol, args.timeframe, report_type='prediction'
                        )
                        if script_path:
                            if self.output_generator.open_terminal(script_path):
                                self.print_success("Predictions opened in new terminal window.")
                            else:
                                self.print_error("Failed to open terminal window.")
                        else:
                            self.print_error("Failed to generate predictions output.")
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

            args = self.build_command_args()

            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)

            if df is not None:
                self.print_info("Performing technical analysis...")
                df_analyzed = analyze_data(df, args)

                if df_analyzed is not None and not df_analyzed.empty:
                    self.print_info("Making predictions...")
                    df_predictions = predict(df_analyzed, args.model_path, args)

                    if df_predictions is not None:
                        self.print_info("Backtesting predictions...")
                        performance_metrics, trading_metrics = backtest(df_predictions, args)

                        if performance_metrics:
                            self.print_success("Performance metrics:")
                            for key, value in performance_metrics.items():
                                print(f"  {key}: {value}")

                        if trading_metrics:
                            self.print_success("\nTrading metrics:")
                            for key, value in trading_metrics.items():
                                print(f"  {key}: {value}")

                        # Generate and open terminal output
                        self.print_info("Opening backtest results in new terminal window...")
                        script_path = self.output_generator.generate_output(
                            df_predictions, args.symbol, args.timeframe, report_type='backtest',
                            performance_metrics=performance_metrics, trading_metrics=trading_metrics
                        )
                        if script_path:
                            if self.output_generator.open_terminal(script_path):
                                self.print_success("Backtest results opened in new terminal window.")
                            else:
                                self.print_error("Failed to open terminal window.")
                        else:
                            self.print_error("Failed to generate backtest output.")
                    else:
                        self.print_error("Failed to make predictions.")
                else:
                    self.print_error("Analysis resulted in empty DataFrame or failed.")
            else:
                self.print_error("Failed to fetch data.")

        except Exception as e:
            self.print_error(f"Error backtesting strategy: {e}")

        self.wait_for_key()

    def run_all_steps(self):
        """Run all steps based on current settings."""
        self.print_header("Running All Steps")

        try:
            from main import fetch_data, analyze_data, train_model, predict, backtest

            args = self.build_command_args()

            # Fetch data
            self.print_info(f"Fetching data for {args.symbol} from {args.exchange}...")
            df = fetch_data(args)
            if df is None:
                self.print_error("Failed to fetch data.")
                self.wait_for_key()
                return

            self.print_success(f"Successfully fetched {len(df)} rows of data.")

            # Analyze data
            self.print_info("Performing technical analysis...")
            df_analyzed = analyze_data(df, args)
            if df_analyzed is None or df_analyzed.empty:
                self.print_error("Analysis resulted in empty DataFrame or failed.")
                self.wait_for_key()
                return

            self.print_success(f"Successfully analyzed data with {len(df_analyzed.columns)} indicators.")

            # Train model
            self.print_info(f"Training {args.model_type} model...")
            model, model_path = train_model(df_analyzed, args)
            if model is None:
                self.print_error("Failed to train model.")
                self.wait_for_key()
                return

            self.print_success("Successfully trained model.")

            if model_path:
                self.print_success(f"Model saved to {model_path}")
                self.settings['model_path'] = model_path
                self.refresh_available_models()

            # Make predictions
            self.print_info("Making predictions...")
            df_predictions = predict(df_analyzed, model_path, args)
            if df_predictions is None:
                self.print_error("Failed to make predictions.")
                self.wait_for_key()
                return

            self.print_success("Successfully made predictions.")

            # Display sample predictions
            print("\nSample predictions:")
            if 'prediction' in df_predictions.columns and 'prediction_probability' in df_predictions.columns:
                print(df_predictions[['close', 'prediction', 'prediction_probability']].tail().to_string())
            else:
                print(df_predictions[['close']].tail().to_string())

            # Backtest
            self.print_info("Backtesting predictions...")
            performance_metrics, trading_metrics = backtest(df_predictions, args)

            if performance_metrics:
                self.print_success("\nPerformance metrics:")
                for key, value in performance_metrics.items():
                    print(f"  {key}: {value}")

            if trading_metrics:
                self.print_success("\nTrading metrics:")
                for key, value in trading_metrics.items():
                    print(f"  {key}: {value}")

            # Generate and open terminal output
            self.print_info("\nOpening complete analysis in new terminal window...")
            script_path = self.output_generator.generate_output(
                df_predictions, args.symbol, args.timeframe, report_type='all',
                performance_metrics=performance_metrics, trading_metrics=trading_metrics
            )
            if script_path:
                if self.output_generator.open_terminal(script_path):
                    self.print_success("Complete analysis opened in new terminal window.")
                else:
                    self.print_error("Failed to open terminal window.")
            else:
                self.print_error("Failed to generate analysis output.")

            self.print_success("\nSuccessfully completed all operations.")

        except Exception as e:
            self.print_error(f"Error running all steps: {e}")

        self.wait_for_key()

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
