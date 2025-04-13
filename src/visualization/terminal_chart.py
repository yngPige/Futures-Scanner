"""
Terminal Chart Generator Module

This module provides functionality to display charts in a terminal window using Plotext.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import plotext as plt

logger = logging.getLogger(__name__)

class TerminalChartGenerator:
    """Class to generate terminal-based charts for cryptocurrency data and technical analysis."""

    def __init__(self, theme='dark'):
        """
        Initialize the TerminalChartGenerator.

        Args:
            theme (str): Chart theme ('dark' or 'light')
        """
        self.theme = theme
        logger.info(f"Initialized TerminalChartGenerator with {theme} theme")

        # Set the theme in plotext
        if theme == 'dark':
            plt.theme('dark')
        else:
            plt.theme('clear')

    def create_price_chart(self, df, title='Price Chart', include_volume=True):
        """
        Create a price chart using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            title (str): Chart title
            include_volume (bool): Whether to include volume

        Returns:
            bool: True if chart was displayed successfully
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create chart")
            return False

        try:
            # Convert datetime index to string format
            dates = [d.strftime('%d/%m/%Y') for d in df.index]

            # Set date format
            plt.date_form('d/m/Y')

            # Clear the previous plot
            plt.clf()

            # Set the title and labels
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price")

            # Plot the price data
            plt.plot(dates, df['close'], label="Close Price")

            # Add SMA if available
            if 'sma_20' in df.columns:
                plt.plot(dates, df['sma_20'], label="SMA 20", color="blue")

            if 'sma_50' in df.columns:
                plt.plot(dates, df['sma_50'], label="SMA 50", color="green")

            # Add legend
            plt.legend()

            # Show the plot
            plt.show()

            logger.info("Terminal price chart displayed successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating terminal price chart: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_candlestick_chart(self, df, title='Candlestick Chart'):
        """
        Create a candlestick chart using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            title (str): Chart title

        Returns:
            bool: True if chart was displayed successfully
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create candlestick chart")
            return False

        try:
            # Convert datetime index to string format
            dates = [d.strftime('%d/%m/%Y') for d in df.index]

            # Set date format
            plt.date_form('d/m/Y')

            # Clear the previous plot
            plt.clf()

            # Set the title and labels
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price")

            # Create a dictionary with OHLC data
            data = {
                'Open': df['open'].tolist(),
                'High': df['high'].tolist(),
                'Low': df['low'].tolist(),
                'Close': df['close'].tolist()
            }

            # Plot the candlestick chart
            plt.candlestick(dates, data)

            # Show the plot
            plt.show()

            logger.info("Terminal candlestick chart displayed successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating terminal candlestick chart: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_advanced_chart_with_suggestions(self, df, llm_analysis=None, title='Advanced Trading Chart'):
        """
        Create an advanced chart with entry/exit suggestions using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            llm_analysis (dict, optional): LLM analysis results with trading recommendations
            title (str): Chart title

        Returns:
            bool: True if chart was displayed successfully
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create advanced chart")
            return False

        try:
            # Convert datetime index to string format
            dates = [d.strftime('%d/%m/%Y') for d in df.index]

            # Set date format
            plt.date_form('d/m/Y')

            # Clear the previous plot
            plt.clf()

            # Set the title and labels
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price")

            # Create a dictionary with OHLC data
            data = {
                'Open': df['open'].tolist(),
                'High': df['high'].tolist(),
                'Low': df['low'].tolist(),
                'Close': df['close'].tolist()
            }

            # Plot the candlestick chart
            plt.candlestick(dates, data)

            # Add entry/exit suggestions if available
            if llm_analysis and isinstance(llm_analysis, dict) and 'error' not in llm_analysis:
                # Get the latest price
                latest_price = df['close'].iloc[-1]

                # Extract entry/exit levels from LLM analysis
                entry_price = llm_analysis.get('entry_price')
                stop_loss = llm_analysis.get('stop_loss')
                take_profit = llm_analysis.get('take_profit')
                recommendation = llm_analysis.get('recommendation', 'NEUTRAL')
                risk = llm_analysis.get('risk', 'MEDIUM')

                # Add horizontal lines for entry, stop loss, and take profit
                if entry_price and isinstance(entry_price, (int, float)):
                    plt.hline(entry_price, color="yellow")
                    plt.text(dates[0], entry_price, f"Entry: {entry_price:.2f}")

                if stop_loss and isinstance(stop_loss, (int, float)):
                    plt.hline(stop_loss, color="red")
                    plt.text(dates[0], stop_loss, f"Stop: {stop_loss:.2f}")

                if take_profit and isinstance(take_profit, (int, float)):
                    plt.hline(take_profit, color="green")
                    plt.text(dates[0], take_profit, f"Target: {take_profit:.2f}")

                # Add recommendation and risk as text
                plt.text(dates[0], df['high'].max(), f"Recommendation: {recommendation} | Risk: {risk}")

                # Calculate risk-reward ratio if both stop loss and take profit are available
                if stop_loss and take_profit and entry_price and isinstance(stop_loss, (int, float)) and isinstance(take_profit, (int, float)) and isinstance(entry_price, (int, float)):
                    risk_amount = abs(entry_price - stop_loss)
                    reward_amount = abs(take_profit - entry_price)
                    if risk_amount > 0:
                        risk_reward_ratio = reward_amount / risk_amount
                        plt.text(dates[len(dates) // 2], df['low'].min(), f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")

            # Add legend
            plt.legend()

            # Show the plot
            plt.show()

            logger.info("Terminal advanced chart displayed successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating terminal advanced chart: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_indicator_chart(self, df, indicator_name, title=None):
        """
        Create a chart for a specific indicator using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with indicator data
            indicator_name (str): Name of the indicator to plot
            title (str, optional): Chart title

        Returns:
            bool: True if chart was displayed successfully
        """
        if df.empty or indicator_name not in df.columns:
            logger.warning(f"Cannot create indicator chart: DataFrame is empty or {indicator_name} not found")
            return False

        try:
            # Convert datetime index to string format
            dates = [d.strftime('%d/%m/%Y') for d in df.index]

            # Set date format
            plt.date_form('d/m/Y')

            # Clear the previous plot
            plt.clf()

            # Set the title and labels
            if title:
                plt.title(title)
            else:
                plt.title(f"{indicator_name} Chart")

            plt.xlabel("Date")
            plt.ylabel(indicator_name)

            # Plot the indicator
            plt.plot(dates, df[indicator_name], label=indicator_name)

            # Add horizontal lines for common indicators
            if indicator_name.lower().startswith('rsi'):
                plt.hline(70, color="red")
                plt.text(dates[0], 70, "Overbought (70)")
                plt.hline(30, color="green")
                plt.text(dates[0], 30, "Oversold (30)")

            # Add legend
            plt.legend()

            # Show the plot
            plt.show()

            logger.info(f"Terminal {indicator_name} chart displayed successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating terminal indicator chart: {e}")
            import traceback
            traceback.print_exc()
            return False
