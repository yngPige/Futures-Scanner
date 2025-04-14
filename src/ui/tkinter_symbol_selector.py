"""
Tkinter Symbol Selector Module for Crypto Futures Scanner

This module provides a GUI window for selecting cryptocurrency symbols
with search functionality and exchange selection dropdown.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
import logging
# No need for colorama in the Tkinter UI
from src.data.data_fetcher import DataFetcher

class TkinterSymbolSelector:
    """Class to provide a Tkinter GUI for selecting cryptocurrency symbols."""

    def __init__(self, current_symbol='BTC/USDT', current_exchange='kraken'):
        """
        Initialize the Tkinter Symbol Selector.

        Args:
            current_symbol (str): The currently selected symbol
            current_exchange (str): The currently selected exchange
        """
        self.current_symbol = current_symbol
        self.current_exchange = current_exchange
        self.available_exchanges = ['CCXT:ALL', 'kraken', 'kucoin', 'huobi']
        self.exchange_symbols = {}
        self.settings_file = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "settings.json")

        # Base cryptocurrencies for generating symbols
        self.base_cryptocurrencies = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC',
            'LINK', 'UNI', 'LTC', 'SHIB', 'TRX', 'ETC', 'TON', 'NEAR', 'BCH', 'ATOM',
            'APE', 'ARB', 'AAVE', 'ALGO', 'APT', 'BAT', 'COMP', 'CRV', 'DASH', 'ENJ',
            'FIL', 'GALA', 'GRT', 'ICP', 'IMX', 'INJ', 'KAVA', 'KSM', 'LDO', 'MANA',
            'MASK', 'MKR', 'NEO', 'OP', 'PEPE', 'RUNE', 'SAND', 'SNX', 'SUI', 'SUSHI',
            'THETA', 'VET', 'WAVES', 'XLM', 'XMR', 'XTZ', 'YFI', 'ZEC', 'ZIL', '1INCH'
        ]

        # Quote currencies and formats - only USDT pairs
        self.quote_formats = [
            {'quote': 'USDT', 'format': '{}/{}'}  # Format for USDT pairs (e.g., BTC/USDT)
        ]

        # Create the main window
        self.root = tk.Tk()
        self.root.title("USDT Pairs Selector")
        self.root.geometry("500x600")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme

        # Configure colors
        self.style.configure("TFrame", background="#2E2E2E")
        self.style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF")
        self.style.configure("TButton", background="#3E3E3E", foreground="#FFFFFF")
        self.style.configure("TCombobox", fieldbackground="#3E3E3E", background="#3E3E3E", foreground="#FFFFFF")
        self.style.map('TCombobox', fieldbackground=[('readonly', '#3E3E3E')])
        self.style.map('TCombobox', selectbackground=[('readonly', '#3E3E3E')])
        self.style.map('TCombobox', selectforeground=[('readonly', '#FFFFFF')])

        self.root.configure(bg="#2E2E2E")

        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create the exchange selection frame
        self.exchange_frame = ttk.Frame(self.main_frame)
        self.exchange_frame.pack(fill=tk.X, pady=5)

        # Exchange label
        ttk.Label(self.exchange_frame, text="Exchange:").pack(side=tk.LEFT, padx=5)

        # Exchange dropdown
        self.exchange_var = tk.StringVar(value=self.current_exchange)
        self.exchange_dropdown = ttk.Combobox(
            self.exchange_frame,
            textvariable=self.exchange_var,
            values=self.available_exchanges,
            state="readonly",
            width=15
        )
        self.exchange_dropdown.pack(side=tk.LEFT, padx=5)
        self.exchange_dropdown.bind("<<ComboboxSelected>>", self.on_exchange_changed)

        # Create the search frame
        self.search_frame = ttk.Frame(self.main_frame)
        self.search_frame.pack(fill=tk.X, pady=5)

        # Search label
        ttk.Label(self.search_frame, text="Search:").pack(side=tk.LEFT, padx=5)

        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.search_var.trace_add("write", self.on_search_changed)

        # Create the symbols frame
        self.symbols_frame = ttk.Frame(self.main_frame)
        self.symbols_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Symbols listbox with scrollbar
        self.symbols_listbox_frame = ttk.Frame(self.symbols_frame)
        self.symbols_listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.symbols_listbox = tk.Listbox(
            self.symbols_listbox_frame,
            selectmode=tk.SINGLE,
            bg="#3E3E3E",
            fg="#FFFFFF",
            selectbackground="#5E5E5E",
            font=("Arial", 10)
        )
        self.symbols_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.symbols_scrollbar = ttk.Scrollbar(
            self.symbols_listbox_frame,
            orient=tk.VERTICAL,
            command=self.symbols_listbox.yview
        )
        self.symbols_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.symbols_listbox.config(yscrollcommand=self.symbols_scrollbar.set)

        # Double-click binding for symbol selection
        self.symbols_listbox.bind("<Double-1>", self.on_symbol_selected)

        # Create the buttons frame
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(fill=tk.X, pady=10)

        # Select button
        self.select_button = ttk.Button(
            self.buttons_frame,
            text="Select",
            command=self.on_select
        )
        self.select_button.pack(side=tk.RIGHT, padx=5)

        # Cancel button
        self.cancel_button = ttk.Button(
            self.buttons_frame,
            text="Cancel",
            command=self.on_cancel
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=5)

        # Status frame
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_var,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)

        # Load symbols
        self.load_settings()
        self.load_symbols()

        # Result variables
        self.result_symbol = None
        self.result_exchange = None
        self.cancelled = True

    def load_settings(self):
        """Load saved settings if they exist."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)

            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

                    if 'symbol' in settings:
                        self.current_symbol = settings['symbol']

                    if 'exchange' in settings and settings['exchange'] in self.available_exchanges:
                        self.current_exchange = settings['exchange']
                        self.exchange_var.set(self.current_exchange)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        """Save current settings."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)

            settings = {
                'symbol': self.current_symbol,
                'exchange': self.current_exchange
            }

            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_symbols(self):
        """Load symbols for the current exchange."""
        self.status_var.set(f"Loading USDT pairs for {self.current_exchange}...")
        self.root.update()

        # Clear the listbox
        self.symbols_listbox.delete(0, tk.END)

        # Start loading exchange symbols immediately
        self.load_exchange_symbols()

        # If we already have symbols for this exchange, display them
        if self.current_exchange in self.exchange_symbols and self.exchange_symbols[self.current_exchange]:
            self.display_available_symbols()
        else:
            # Otherwise, show a loading message in the listbox
            self.symbols_listbox.insert(tk.END, "Loading available pairs...")
            # And schedule a check to update when symbols are loaded
            self.root.after(500, self.check_and_display_symbols)

    def load_exchange_symbols(self):
        """Load exchange-specific symbols in the background."""
        if self.current_exchange in self.exchange_symbols:
            # Already loaded
            self.status_var.set(f"Loaded {len(self.exchange_symbols[self.current_exchange])} USDT pairs from {self.current_exchange}")
            return

        # Temporarily disable INFO logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        try:
            # Try to get both spot and futures symbols
            spot_fetcher = DataFetcher(exchange_id=self.current_exchange, market_type='spot')
            futures_fetcher = DataFetcher(exchange_id=self.current_exchange, market_type='future')

            # Get only USDT pairs from both markets
            spot_symbols = spot_fetcher.get_available_symbols(quote_currency='USDT')
            futures_symbols = futures_fetcher.get_available_symbols(quote_currency='USDT')

            # Combine symbols (remove duplicates)
            all_symbols = list(set(spot_symbols + futures_symbols))

            # Store the symbols
            self.exchange_symbols[self.current_exchange] = all_symbols

            # Update the display with the loaded symbols
            self.display_available_symbols()

            self.status_var.set(f"Loaded {len(all_symbols)} USDT pairs from {self.current_exchange}")
        except Exception as e:
            self.exchange_symbols[self.current_exchange] = []
            self.status_var.set(f"Error loading USDT pairs from {self.current_exchange}: {e}")
        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

    def check_and_display_symbols(self):
        """Check if symbols are loaded and display them if they are."""
        if self.current_exchange in self.exchange_symbols and self.exchange_symbols[self.current_exchange]:
            # Symbols are loaded, display them
            self.display_available_symbols()
        else:
            # Symbols are still loading, check again later
            self.root.after(500, self.check_and_display_symbols)

    def display_available_symbols(self):
        """Display only the available symbols for the current exchange."""
        if self.current_exchange not in self.exchange_symbols:
            return

        # Get available symbols
        available_symbols = self.exchange_symbols[self.current_exchange]

        # Get current search term
        search_term = self.search_var.get().upper()

        # Clear the listbox
        self.symbols_listbox.delete(0, tk.END)

        # Filter symbols by search term if needed
        if search_term:
            filtered_symbols = [s for s in available_symbols if search_term in s]
        else:
            filtered_symbols = available_symbols

        # Add all available symbols to the listbox
        for symbol in sorted(filtered_symbols):
            self.symbols_listbox.insert(tk.END, symbol)

        # Try to select the current symbol if it's available
        try:
            index = filtered_symbols.index(self.current_symbol)
            self.symbols_listbox.selection_set(index)
            self.symbols_listbox.see(index)
        except ValueError:
            # Current symbol not available, try to select the first item
            if self.symbols_listbox.size() > 0:
                self.symbols_listbox.selection_set(0)
                self.symbols_listbox.see(0)

    def filter_symbols(self):
        """Filter symbols based on search term."""
        # If we have symbols for this exchange, display them with the current filter
        if self.current_exchange in self.exchange_symbols and self.exchange_symbols[self.current_exchange]:
            self.display_available_symbols()
        else:
            # Otherwise, we're still loading - show a message
            self.symbols_listbox.delete(0, tk.END)
            self.symbols_listbox.insert(tk.END, "Loading available pairs...")

    def on_search_changed(self, *_):
        """Handle search term changes."""
        self.filter_symbols()

    def on_exchange_changed(self, _):
        """Handle exchange selection changes."""
        self.current_exchange = self.exchange_var.get()

        # Clear the search field when changing exchanges
        self.search_var.set("")

        # Load symbols for the new exchange
        self.load_symbols()

    def on_symbol_selected(self, _):
        """Handle double-click on a symbol."""
        self.on_select()

    def on_select(self):
        """Handle select button click."""
        selection = self.symbols_listbox.curselection()

        if not selection:
            messagebox.showwarning("No Selection", "Please select a symbol first.")
            return

        # Get the selected symbol (which is guaranteed to be available)
        selected_symbol = self.symbols_listbox.get(selection[0])

        # If we're still loading, don't allow selection
        if selected_symbol == "Loading available pairs...":
            messagebox.showinfo("Loading", "Still loading available pairs. Please wait a moment.")
            return

        # Set result variables
        self.result_symbol = selected_symbol
        self.result_exchange = self.current_exchange
        self.cancelled = False

        # Save settings
        self.current_symbol = selected_symbol
        self.save_settings()

        # Close the window
        self.root.destroy()

    def on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.root.destroy()

    def on_close(self):
        """Handle window close."""
        self.cancelled = True
        self.root.destroy()

    # We no longer need the find_alternative_symbol method since we only show available symbols

    def run(self):
        """
        Run the Tkinter symbol selector.

        Returns:
            tuple: (selected_symbol, exchange, cancelled)
                selected_symbol (str): The selected symbol
                exchange (str): The selected exchange
                cancelled (bool): True if the selection was cancelled
        """
        # Center the window on the screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Run the main loop
        self.root.mainloop()

        return self.result_symbol, self.result_exchange, self.cancelled


def select_symbol_tkinter(current_symbol='BTC/USDT', current_exchange='kraken'):
    """
    Show a Tkinter window for selecting a cryptocurrency symbol.

    Args:
        current_symbol (str): The currently selected symbol
        current_exchange (str): The currently selected exchange

    Returns:
        tuple: (selected_symbol, exchange, cancelled)
            selected_symbol (str): The selected symbol
            exchange (str): The selected exchange
            cancelled (bool): True if the selection was cancelled
    """
    selector = TkinterSymbolSelector(current_symbol, current_exchange)
    return selector.run()


# Example usage
if __name__ == "__main__":
    symbol, exchange, cancelled = select_symbol_tkinter()

    if cancelled:
        print("Selection cancelled")
    else:
        print(f"Selected symbol: {symbol} on {exchange}")
