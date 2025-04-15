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
        self.favorites = []  # List to store favorite symbols

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

        # Notification label for favorites
        self.notification_var = tk.StringVar(value="")
        self.notification_frame = ttk.Frame(self.exchange_frame)
        self.notification_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.notification_label = tk.Label(
            self.notification_frame,
            textvariable=self.notification_var,
            fg="#FFFFFF",  # White text
            bg="#4CAF50",  # Green background
            font=("Arial", 10, "bold"),
            padx=8,
            pady=2,
            borderwidth=1,
            relief="flat"
        )
        # Don't pack the label yet - it will only be shown when there's a notification

        # Create the search frame
        self.search_frame = ttk.Frame(self.main_frame)
        self.search_frame.pack(fill=tk.X, pady=5)

        # Search label
        ttk.Label(self.search_frame, text="Search:").pack(side=tk.LEFT, padx=5)

        # Search entry with placeholder text
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.search_var.trace_add("write", self.on_search_changed)

        # Add focus/blur events to handle placeholder text
        self.search_entry.bind("<FocusIn>", self.on_search_focus_in)
        self.search_entry.bind("<FocusOut>", self.on_search_focus_out)

        # Set initial placeholder text
        self.search_entry.insert(0, "Type to search...")
        self.search_entry.config(foreground="gray")

        # Create the notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create the All Symbols tab
        self.all_symbols_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.all_symbols_frame, text="All Symbols")

        # Create the Favorites tab
        self.favorites_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.favorites_frame, text="Favorites")

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # All Symbols listbox with scrollbar
        self.all_symbols_listbox_frame = ttk.Frame(self.all_symbols_frame)
        self.all_symbols_listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.all_symbols_listbox = tk.Listbox(
            self.all_symbols_listbox_frame,
            selectmode=tk.SINGLE,
            bg="#3E3E3E",
            fg="#FFFFFF",
            selectbackground="#5E5E5E",
            font=("Arial", 10)
        )
        self.all_symbols_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.all_symbols_scrollbar = ttk.Scrollbar(
            self.all_symbols_listbox_frame,
            orient=tk.VERTICAL,
            command=self.all_symbols_listbox.yview
        )
        self.all_symbols_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.all_symbols_listbox.config(yscrollcommand=self.all_symbols_scrollbar.set)

        # Double-click binding for symbol selection in All Symbols tab
        self.all_symbols_listbox.bind("<Double-1>", self.on_symbol_selected)

        # Favorites listbox with scrollbar
        self.favorites_listbox_frame = ttk.Frame(self.favorites_frame)
        self.favorites_listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.favorites_listbox = tk.Listbox(
            self.favorites_listbox_frame,
            selectmode=tk.SINGLE,
            bg="#3E3E3E",
            fg="#FFFFFF",
            selectbackground="#5E5E5E",
            font=("Arial", 10)
        )
        self.favorites_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.favorites_scrollbar = ttk.Scrollbar(
            self.favorites_listbox_frame,
            orient=tk.VERTICAL,
            command=self.favorites_listbox.yview
        )
        self.favorites_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.favorites_listbox.config(yscrollcommand=self.favorites_scrollbar.set)

        # Double-click binding for symbol selection in Favorites tab
        self.favorites_listbox.bind("<Double-1>", self.on_symbol_selected)

        # Store the active listbox reference
        self.active_listbox = self.all_symbols_listbox

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

        # Favorite button
        self.favorite_button = ttk.Button(
            self.buttons_frame,
            text="★ Favorite",
            command=self.toggle_favorite
        )
        self.favorite_button.pack(side=tk.LEFT, padx=5)

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

                    # Load favorites
                    if 'favorites' in settings and isinstance(settings['favorites'], list):
                        self.favorites = settings['favorites']
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        """Save current settings."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)

            settings = {
                'symbol': self.current_symbol,
                'exchange': self.current_exchange,
                'favorites': self.favorites
            }

            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_symbols(self):
        """Load symbols for the current exchange."""
        self.status_var.set(f"Loading USDT pairs for {self.current_exchange}...")
        self.root.update()

        # Clear both listboxes
        self.all_symbols_listbox.delete(0, tk.END)
        self.favorites_listbox.delete(0, tk.END)

        # Start loading exchange symbols immediately
        self.load_exchange_symbols()

        # If we already have symbols for this exchange, display them
        if self.current_exchange in self.exchange_symbols and self.exchange_symbols[self.current_exchange]:
            self.display_available_symbols()
            self.display_favorite_symbols()
        else:
            # Otherwise, show a loading message in the listboxes
            self.all_symbols_listbox.insert(tk.END, "Loading available pairs...")
            self.favorites_listbox.insert(tk.END, "Loading available pairs...")
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
            self.display_favorite_symbols()
        else:
            # Symbols are still loading, check again later
            self.root.after(500, self.check_and_display_symbols)

    def on_tab_changed(self, _):
        """Handle tab change event."""
        selected_tab = self.notebook.index(self.notebook.select())

        # Update the active listbox reference
        if selected_tab == 0:  # All Symbols tab
            self.active_listbox = self.all_symbols_listbox
        else:  # Favorites tab
            self.active_listbox = self.favorites_listbox

    def display_available_symbols(self):
        """Display all available symbols for the current exchange in the All Symbols tab."""
        if self.current_exchange not in self.exchange_symbols:
            return

        # Get available symbols
        available_symbols = self.exchange_symbols[self.current_exchange]

        # Get current search term
        search_term = self.search_var.get().upper()

        # Clear the All Symbols listbox
        self.all_symbols_listbox.delete(0, tk.END)

        # Filter symbols by search term if needed
        if search_term:
            filtered_symbols = [s for s in available_symbols if search_term in s]
        else:
            filtered_symbols = available_symbols

        # Add all available symbols to the All Symbols listbox
        for symbol in sorted(filtered_symbols):
            # Add a star prefix for favorite symbols
            if symbol in self.favorites:
                display_text = f"★ {symbol}"
            else:
                display_text = f"   {symbol}"

            self.all_symbols_listbox.insert(tk.END, display_text)

        # Try to select the current symbol if it's available
        try:
            # Find the index of the current symbol in the listbox
            for i in range(self.all_symbols_listbox.size()):
                item_text = self.all_symbols_listbox.get(i)
                if self.current_symbol in item_text:  # Check if symbol is in the display text
                    self.all_symbols_listbox.selection_set(i)
                    self.all_symbols_listbox.see(i)
                    break
        except Exception:
            # Current symbol not available, try to select the first item
            if self.all_symbols_listbox.size() > 0:
                self.all_symbols_listbox.selection_set(0)
                self.all_symbols_listbox.see(0)

    def display_favorite_symbols(self):
        """Display favorite symbols for the current exchange in the Favorites tab."""
        if self.current_exchange not in self.exchange_symbols:
            return

        # Get available symbols
        available_symbols = self.exchange_symbols[self.current_exchange]

        # Get current search term
        search_term = self.search_var.get().upper()

        # Clear the Favorites listbox
        self.favorites_listbox.delete(0, tk.END)

        # Filter symbols by search term and favorites
        filtered_symbols = [s for s in available_symbols if s in self.favorites]
        if search_term:
            filtered_symbols = [s for s in filtered_symbols if search_term in s]

        # Add favorite symbols to the Favorites listbox
        for symbol in sorted(filtered_symbols):
            display_text = f"★ {symbol}"
            self.favorites_listbox.insert(tk.END, display_text)

        # Try to select the current symbol if it's a favorite
        if self.current_symbol in self.favorites:
            try:
                # Find the index of the current symbol in the listbox
                for i in range(self.favorites_listbox.size()):
                    item_text = self.favorites_listbox.get(i)
                    if self.current_symbol in item_text:  # Check if symbol is in the display text
                        self.favorites_listbox.selection_set(i)
                        self.favorites_listbox.see(i)
                        break
            except Exception:
                # Current symbol not available, try to select the first item
                if self.favorites_listbox.size() > 0:
                    self.favorites_listbox.selection_set(0)
                    self.favorites_listbox.see(0)

    def filter_symbols(self):
        """Filter symbols based on search term."""
        # If we have symbols for this exchange, display them with the current filter
        if self.current_exchange in self.exchange_symbols and self.exchange_symbols[self.current_exchange]:
            self.display_available_symbols()
            self.display_favorite_symbols()
        else:
            # Otherwise, we're still loading - show a message
            self.all_symbols_listbox.delete(0, tk.END)
            self.all_symbols_listbox.insert(tk.END, "Loading available pairs...")
            self.favorites_listbox.delete(0, tk.END)
            self.favorites_listbox.insert(tk.END, "Loading available pairs...")

    def on_search_focus_in(self, _):
        """Handle search entry focus in event."""
        if self.search_var.get() == "Type to search...":
            self.search_entry.delete(0, tk.END)
            self.search_entry.config(foreground="white")

    def on_search_focus_out(self, _):
        """Handle search entry focus out event."""
        if not self.search_var.get():
            self.search_entry.insert(0, "Type to search...")
            self.search_entry.config(foreground="gray")

    def on_search_changed(self, *_):
        """Handle search term changes."""
        # Skip filtering if the text is the placeholder
        if self.search_var.get() != "Type to search...":
            self.filter_symbols()

    def on_exchange_changed(self, _):
        """Handle exchange selection changes."""
        self.current_exchange = self.exchange_var.get()

        # Clear the search field when changing exchanges
        self.search_var.set("")

        # Load symbols for the new exchange
        self.load_symbols()

    def on_symbol_selected(self, event):
        """Handle double-click on a symbol."""
        # Determine which listbox was clicked
        if event.widget == self.all_symbols_listbox:
            self.active_listbox = self.all_symbols_listbox
        else:  # event.widget == self.favorites_listbox
            self.active_listbox = self.favorites_listbox

        self.on_select()

    def on_select(self):
        """Handle select button click."""
        selection = self.active_listbox.curselection()

        if not selection:
            messagebox.showwarning("No Selection", "Please select a symbol first.")
            return

        # Get the selected symbol (which is guaranteed to be available)
        selected_display_text = self.active_listbox.get(selection[0])

        # If we're still loading, don't allow selection
        if selected_display_text == "Loading available pairs...":
            messagebox.showinfo("Loading", "Still loading available pairs. Please wait a moment.")
            return

        # Extract the actual symbol from the display text (remove star if present)
        selected_symbol = selected_display_text.strip().replace('★ ', '')

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

    def toggle_favorite(self):
        """Toggle favorite status for the selected symbol."""
        selection = self.active_listbox.curselection()

        if not selection:
            messagebox.showwarning("No Selection", "Please select a symbol first.")
            return

        # Get the selected symbol display text
        selected_display_text = self.active_listbox.get(selection[0])

        # If we're still loading, don't allow favoriting
        if selected_display_text == "Loading available pairs...":
            messagebox.showinfo("Loading", "Still loading available pairs. Please wait a moment.")
            return

        # Extract the actual symbol from the display text (remove star if present)
        selected_symbol = selected_display_text.strip().replace('★ ', '')

        # Toggle favorite status
        if selected_symbol in self.favorites:
            self.favorites.remove(selected_symbol)
            self.show_notification(f"Removed {selected_symbol} from favorites")
        else:
            self.favorites.append(selected_symbol)
            self.show_notification(f"Added {selected_symbol} to favorites")

        # Save settings
        self.save_settings()

        # Refresh the display
        self.display_available_symbols()
        self.display_favorite_symbols()

        # If we're in the favorites tab and removed a favorite, select another item if available
        if self.active_listbox == self.favorites_listbox and selected_symbol not in self.favorites:
            if self.favorites_listbox.size() > 0:
                self.favorites_listbox.selection_set(0)
                self.favorites_listbox.see(0)

    def show_notification(self, message):
        """Show a notification message and clear it after a delay."""
        # Set the notification text
        self.notification_var.set(message)

        # Determine color based on message content
        if "Added" in message:
            self.notification_label.config(bg="#4CAF50")  # Green for adding
        else:  # "Removed" in message
            self.notification_label.config(bg="#F44336")  # Red for removing

        # Show the notification label
        self.notification_label.pack(fill=tk.X, expand=True)

        # Schedule clearing the notification after 3 seconds
        self.root.after(3000, self.clear_notification)

    def clear_notification(self):
        """Clear the notification message."""
        self.notification_var.set("")
        # Hide the notification label
        self.notification_label.pack_forget()

    def _set_focus_and_clear_placeholder(self):
        """Set focus to the search entry and clear placeholder text."""
        self.search_entry.focus_set()
        if self.search_var.get() == "Type to search...":
            self.search_entry.delete(0, tk.END)
            self.search_entry.config(foreground="white")

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

        # Set focus to the search entry box and clear placeholder text
        self.root.after(100, self._set_focus_and_clear_placeholder)

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
