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

    def create_candlestick_chart(self, df, title='Candlestick Chart', timeframe=None):
        """
        Create a candlestick chart using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            title (str): Chart title
            timeframe (str, optional): Timeframe of the data (e.g., '1h', '4h', '1d')

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

            # Extract timeframe from title if not provided explicitly
            if not timeframe and ' - ' in title and 'Timeframe' in title:
                title_parts = title.split(' - ')
                if len(title_parts) > 1:
                    timeframe_part = title_parts[1].split(' ')[0]
                    timeframe = timeframe_part

            # Add prediction to the chart if available
            if 'prediction' in df.columns:
                # Get the latest prediction
                latest_pred = df['prediction'].iloc[-1]
                pred_text = "BULLISH" if latest_pred == 1 else "BEARISH"
                pred_color = "green" if latest_pred == 1 else "red"

                # Get prediction probability if available
                pred_prob = 0.5
                if 'prediction_probability' in df.columns:
                    pred_prob = df['prediction_probability'].iloc[-1]

                # Get the latest price for calculating entry, stop loss, and take profit
                latest_price = df['close'].iloc[-1]

                # Calculate entry, stop loss, and take profit based on prediction
                # These are placeholder calculations - they should be replaced with actual model predictions
                if pred_text == "BULLISH":
                    entry_price = latest_price * 0.995  # Slightly below current price
                    stop_loss = entry_price * 0.97     # 3% below entry
                    take_profit = entry_price * 1.05   # 5% above entry
                else:  # BEARISH
                    entry_price = latest_price * 1.005  # Slightly above current price
                    stop_loss = entry_price * 1.03     # 3% above entry
                    take_profit = entry_price * 0.95   # 5% below entry

                # Add prediction text in the upper left corner
                plt.text(dates[0], df['high'].max(), f"PREDICTION: {pred_text} ({pred_prob:.2f})", color=pred_color)

                # Add horizontal lines for entry, stop loss, and take profit
                plt.hline(entry_price, color="yellow")
                plt.text(dates[0], entry_price, f"Entry: {entry_price:.2f}", color="yellow")

                plt.hline(stop_loss, color="red")
                plt.text(dates[0], stop_loss, f"Stop Loss: {stop_loss:.2f}", color="red")

                plt.hline(take_profit, color="green")
                plt.text(dates[0], take_profit, f"Take Profit: {take_profit:.2f}", color="green")

                # Calculate risk-reward ratio
                risk_amount = abs(entry_price - stop_loss)
                reward_amount = abs(take_profit - entry_price)
                if risk_amount > 0:
                    risk_reward_ratio = reward_amount / risk_amount
                    plt.text(dates[0], df['high'].max() * 0.97, f"Risk/Reward: 1:{risk_reward_ratio:.2f}")

            # Add timeframe display in the upper right corner if available
            if timeframe:
                current_time = datetime.now().strftime("%b %d, %Y %H:%M UTC")
                plt.text(dates[-1], df['high'].max(), f"{timeframe} | {current_time} | 3lack_Hands")

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

    def create_advanced_chart_with_suggestions(self, df, llm_analysis=None, title='Advanced Trading Chart', timeframe=None):
        """
        Create an advanced chart with entry/exit suggestions using Plotext.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            llm_analysis (dict, optional): LLM analysis results with trading recommendations
            title (str): Chart title
            timeframe (str, optional): Timeframe of the data (e.g., '1h', '4h', '1d')

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

            # Extract timeframe from title if not provided explicitly
            if not timeframe and ' - ' in title and 'Timeframe' in title:
                title_parts = title.split(' - ')
                if len(title_parts) > 1:
                    timeframe_part = title_parts[1].split(' ')[0]
                    timeframe = timeframe_part

            # Add prediction to the chart if available
            if 'prediction' in df.columns:
                # Get the latest prediction
                latest_pred = df['prediction'].iloc[-1]
                pred_text = "BULLISH" if latest_pred == 1 else "BEARISH"
                pred_color = "green" if latest_pred == 1 else "red"

                # Get prediction probability if available
                pred_prob = 0.5
                if 'prediction_probability' in df.columns:
                    pred_prob = df['prediction_probability'].iloc[-1]

                # Get the latest price for calculating entry, stop loss, and take profit
                latest_price = df['close'].iloc[-1]

                # Calculate entry, stop loss, and take profit based on prediction
                # These are placeholder calculations - they should be replaced with actual model predictions
                if pred_text == "BULLISH":
                    entry_price = latest_price * 0.995  # Slightly below current price
                    stop_loss = entry_price * 0.97     # 3% below entry
                    take_profit = entry_price * 1.05   # 5% above entry
                else:  # BEARISH
                    entry_price = latest_price * 1.005  # Slightly above current price
                    stop_loss = entry_price * 1.03     # 3% above entry
                    take_profit = entry_price * 0.95   # 5% below entry

                # Add prediction text in the upper left corner
                plt.text(dates[0], df['high'].max(), f"PREDICTION: {pred_text} ({pred_prob:.2f})", color=pred_color)

                # Add horizontal lines for entry, stop loss, and take profit
                plt.hline(entry_price, color="yellow")
                plt.text(dates[0], entry_price, f"Entry: {entry_price:.2f}", color="yellow")

                plt.hline(stop_loss, color="red")
                plt.text(dates[0], stop_loss, f"Stop Loss: {stop_loss:.2f}", color="red")

                plt.hline(take_profit, color="green")
                plt.text(dates[0], take_profit, f"Take Profit: {take_profit:.2f}", color="green")

                # Calculate risk-reward ratio
                risk_amount = abs(entry_price - stop_loss)
                reward_amount = abs(take_profit - entry_price)
                if risk_amount > 0:
                    risk_reward_ratio = reward_amount / risk_amount
                    plt.text(dates[0], df['high'].max() * 0.97, f"Risk/Reward: 1:{risk_reward_ratio:.2f}")

            # Add timeframe display in the upper right corner if available
            if timeframe:
                current_time = datetime.now().strftime("%b %d, %Y %H:%M UTC")
                plt.text(dates[-1], df['high'].max(), f"{timeframe} | {current_time} | 3lack_Hands")

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

                # Extract sentiment and risk from LLM analysis
                sentiment = llm_analysis.get('sentiment', 'neutral').upper()
                risk = llm_analysis.get('risk', 'medium').upper()

                # Try to extract entry/exit levels from LLM analysis text
                analysis_text = llm_analysis.get('analysis', '')

                # Initialize variables
                entry_price = None
                stop_loss = None
                take_profit = None

                # Try to extract price levels from the analysis text
                try:
                    # Look for support/resistance levels
                    if 'support' in analysis_text.lower():
                        support_idx = analysis_text.lower().find('support')
                        support_text = analysis_text[support_idx:support_idx+100]
                        # Extract numbers from the text
                        import re
                        numbers = re.findall(r'\d+\.\d+', support_text)
                        if numbers:
                            # Use the first number as stop loss for bullish, or entry for bearish
                            if sentiment == 'BULLISH':
                                stop_loss = float(numbers[0])
                            else:
                                entry_price = float(numbers[0])

                    if 'resistance' in analysis_text.lower():
                        resistance_idx = analysis_text.lower().find('resistance')
                        resistance_text = analysis_text[resistance_idx:resistance_idx+100]
                        # Extract numbers from the text
                        import re
                        numbers = re.findall(r'\d+\.\d+', resistance_text)
                        if numbers:
                            # Use the first number as take profit for bullish, or stop loss for bearish
                            if sentiment == 'BULLISH':
                                take_profit = float(numbers[0])
                            else:
                                stop_loss = float(numbers[0])

                    # If we couldn't extract from support/resistance, look for entry/exit points
                    if 'entry' in analysis_text.lower() and not entry_price:
                        entry_idx = analysis_text.lower().find('entry')
                        entry_text = analysis_text[entry_idx:entry_idx+100]
                        # Extract numbers from the text
                        import re
                        numbers = re.findall(r'\d+\.\d+', entry_text)
                        if numbers:
                            entry_price = float(numbers[0])

                    if 'exit' in analysis_text.lower() and not take_profit:
                        exit_idx = analysis_text.lower().find('exit')
                        exit_text = analysis_text[exit_idx:exit_idx+100]
                        # Extract numbers from the text
                        import re
                        numbers = re.findall(r'\d+\.\d+', exit_text)
                        if numbers:
                            take_profit = float(numbers[0])

                    if 'stop' in analysis_text.lower() and not stop_loss:
                        stop_idx = analysis_text.lower().find('stop')
                        stop_text = analysis_text[stop_idx:stop_idx+100]
                        # Extract numbers from the text
                        import re
                        numbers = re.findall(r'\d+\.\d+', stop_text)
                        if numbers:
                            stop_loss = float(numbers[0])
                except Exception as e:
                    logger.warning(f"Error extracting price levels from LLM analysis: {e}")

                # If we still don't have values, use default calculations
                if not entry_price:
                    if sentiment == 'BULLISH':
                        entry_price = latest_price * 0.995  # Slightly below current price
                    else:  # BEARISH or NEUTRAL
                        entry_price = latest_price * 1.005  # Slightly above current price

                if not stop_loss:
                    if sentiment == 'BULLISH':
                        stop_loss = entry_price * 0.97     # 3% below entry
                    else:  # BEARISH or NEUTRAL
                        stop_loss = entry_price * 1.03     # 3% above entry

                if not take_profit:
                    if sentiment == 'BULLISH':
                        take_profit = entry_price * 1.05   # 5% above entry
                    else:  # BEARISH or NEUTRAL
                        take_profit = entry_price * 0.95   # 5% below entry

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

                # Add sentiment and risk as text
                plt.text(dates[0], df['high'].max(), f"Sentiment: {sentiment} | Risk: {risk}")

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
