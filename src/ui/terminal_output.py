"""
Terminal Output Module for Crypto Futures Scanner

This module provides functionality to display analysis results in a new terminal window.
"""

import os
import sys
import subprocess
import tempfile
import logging
import platform
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class TerminalOutputGenerator:
    """Generate and display analysis results in a new terminal window."""

    def __init__(self, theme='dark'):
        """Initialize the terminal output generator."""
        self.theme = theme
        self.temp_dir = tempfile.gettempdir()
        self.system = platform.system()

    def generate_output(self, df, symbol, timeframe, report_type='analysis',
                       performance_metrics=None, trading_metrics=None):
        """
        Generate a terminal output script for the given data.

        Args:
            df (pd.DataFrame): DataFrame with analysis data
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the analysis
            report_type (str): Type of report ('analysis', 'prediction', 'backtest', 'all')
            performance_metrics (dict, optional): Performance metrics from backtest
            trading_metrics (dict, optional): Trading metrics from backtest

        Returns:
            str: Path to the generated script
        """
        try:
            # Create script filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if self.system == 'Windows':
                script_path = os.path.join(self.temp_dir, f"{symbol.replace('/', '_')}_{timeframe}_{report_type}_{timestamp}.bat")
                content = self._generate_windows_batch_content(df, symbol, timeframe, report_type,
                                                             performance_metrics, trading_metrics)
            else:  # Linux or macOS
                script_path = os.path.join(self.temp_dir, f"{symbol.replace('/', '_')}_{timeframe}_{report_type}_{timestamp}.sh")
                content = self._generate_shell_content(df, symbol, timeframe, report_type,
                                                     performance_metrics, trading_metrics)

            # Write script to file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Make the script executable on Unix-like systems
            if self.system != 'Windows':
                os.chmod(script_path, 0o755)

            logger.info(f"Terminal output script generated: {script_path}")
            return script_path

        except Exception as e:
            logger.error(f"Error generating terminal output script: {e}")
            return None

    def _generate_windows_batch_content(self, df, symbol, timeframe, report_type,
                                      performance_metrics, trading_metrics):
        """Generate Windows batch script content."""
        # Get the latest data point
        latest = df.iloc[-1]

        # Start with title and color setup
        content = "@echo off\n"
        content += "title Analysis Overview - " + symbol + " (" + timeframe + ")\n"
        content += "color 0F\n"  # Black background, white text for dark theme
        if self.theme != 'dark':
            content += "color F0\n"  # White background, black text for light theme

        content += "cls\n"
        content += "echo.\n"

        # Add header
        header = f"{report_type.upper()} OVERVIEW - {symbol} ({timeframe})"
        content += f"echo {header}\n"
        content += f"echo {'-' * len(header)}\n"
        content += "echo Generated on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
        content += "echo.\n"

        # Add market overview section
        content += "echo MARKET OVERVIEW\n"
        content += "echo --------------\n"
        content += f"echo Open:   {latest.get('open', 'N/A'):.2f}\n"
        content += f"echo High:   {latest.get('high', 'N/A'):.2f}\n"
        content += f"echo Low:    {latest.get('low', 'N/A'):.2f}\n"
        content += f"echo Close:  {latest.get('close', 'N/A'):.2f}\n"
        content += f"echo Volume: {latest.get('volume', 'N/A'):.0f}\n"
        content += "echo.\n"

        # Add key indicators section
        content += "echo KEY INDICATORS\n"
        content += "echo -------------\n"

        # Add RSI
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            content += f"echo RSI (14): {rsi:.2f} - {rsi_signal}\n"

        # Add EMA Crossover
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            ema_signal = "Bullish" if ema_12 > ema_26 else "Bearish"
            content += f"echo EMA Crossover: {ema_signal} (EMA12: {ema_12:.2f}, EMA26: {ema_26:.2f})\n"

        # Add MACD
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            macd_hist = macd - macd_signal
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            content += f"echo MACD: {macd:.2f} - {macd_trend} (Signal: {macd_signal:.2f}, Hist: {macd_hist:.2f})\n"

        # Add Bollinger Bands
        if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_lower = latest['BBL_20_2.0']
            bb_middle = latest['BBM_20_2.0']
            bb_upper = latest['BBU_20_2.0']
            close = latest['close']
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
            bb_signal = "Lower Band" if close < bb_lower else "Upper Band" if close > bb_upper else "Middle Band"
            content += f"echo Bollinger Bands: {bb_signal} (Position: {bb_position:.1f}%%)\n"

        content += "echo.\n"

        # Add overall signal if available
        if 'signal' in df.columns:
            signal_value = latest['signal']
            signal_text = "Strong Buy" if signal_value > 0.6 else \
                         "Buy" if signal_value > 0 else \
                         "Strong Sell" if signal_value < -0.6 else \
                         "Sell" if signal_value < 0 else "Neutral"
            content += f"echo OVERALL SIGNAL: {signal_text}\n"
            content += "echo.\n"

        # Add prediction if available
        if 'prediction' in df.columns:
            pred_value = latest['prediction']
            pred_prob = latest.get('prediction_probability', 0.5)
            pred_text = "Bullish" if pred_value == 1 else "Bearish"
            content += f"echo PREDICTION: {pred_text} (Confidence: {pred_prob:.2f})\n"
            content += "echo.\n"

        # Add recent signals section
        if 'signal' in df.columns:
            content += "echo RECENT SIGNALS\n"
            content += "echo --------------\n"
            content += "echo Date                    Signal    Close\n"
            content += "echo ---------------------------------------\n"

            # Get recent signals (last 10 periods)
            recent_df = df.tail(10).copy()
            recent_df['date'] = recent_df.index

            # Add signals
            signal_count = 0
            for _, row in recent_df.iterrows():
                if row['signal'] == 1:
                    signal_text = "BUY"
                elif row['signal'] == -1:
                    signal_text = "SELL"
                else:
                    continue  # Skip neutral signals

                date_str = str(row['date'])
                price_str = f"{row['close']:.2f}"
                content += f"echo {date_str}    {signal_text}      {price_str}\n"
                signal_count += 1

            if signal_count == 0:
                content += "echo No recent signals found.\n"

            content += "echo.\n"

        # Add performance metrics if available
        if performance_metrics or trading_metrics:
            content += "echo PERFORMANCE METRICS\n"
            content += "echo -------------------\n"

            if performance_metrics:
                for key, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    content += f"echo {key.replace('_', ' ').title()}: {formatted_value}\n"

            if trading_metrics:
                content += "echo.\n"
                content += "echo TRADING METRICS\n"
                content += "echo --------------\n"
                for key, value in trading_metrics.items():
                    if isinstance(value, (int, float)):
                        if key in ['strategy_return', 'buy_hold_return', 'annualized_strategy_return', 'annualized_buy_hold_return']:
                            formatted_value = f"{value*100:.2f}%%" if not pd.isna(value) else "N/A"
                        elif key in ['win_rate']:
                            formatted_value = f"{value*100:.2f}%%" if not pd.isna(value) else "N/A"
                        else:
                            formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    content += f"echo {key.replace('_', ' ').title()}: {formatted_value}\n"

            content += "echo.\n"

        # Add footer
        content += "echo Press any key to close this window...\n"
        content += "pause > nul\n"

        return content

    def _generate_shell_content(self, df, symbol, timeframe, report_type,
                              performance_metrics, trading_metrics):
        """Generate shell script content for Unix-like systems."""
        # Get the latest data point
        latest = df.iloc[-1]

        # Start with shebang and title
        content = "#!/bin/bash\n\n"

        # Add color definitions
        content += "# Define colors\n"
        content += "RESET='\\033[0m'\n"
        content += "BOLD='\\033[1m'\n"
        content += "RED='\\033[31m'\n"
        content += "GREEN='\\033[32m'\n"
        content += "YELLOW='\\033[33m'\n"
        content += "BLUE='\\033[34m'\n"
        content += "MAGENTA='\\033[35m'\n"
        content += "CYAN='\\033[36m'\n"
        content += "WHITE='\\033[37m'\n\n"

        # Set terminal title
        content += f"echo -ne \"\\033]0;Analysis Overview - {symbol} ({timeframe})\\007\"\n"
        content += "clear\n\n"

        # Add header
        header = f"{report_type.upper()} OVERVIEW - {symbol} ({timeframe})"
        content += f"echo -e \"$BOLD$CYAN{header}$RESET\"\n"
        content += f"echo -e \"$CYAN{'-' * len(header)}$RESET\"\n"
        content += f"echo \"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\"\n"
        content += "echo\n"

        # Add market overview section
        content += "echo -e \"$BOLD$YELLOW MARKET OVERVIEW $RESET\"\n"
        content += "echo -e \"$YELLOW --------------$RESET\"\n"
        content += f"echo -e \"Open:   {latest.get('open', 'N/A'):.2f}\"\n"
        content += f"echo -e \"High:   {latest.get('high', 'N/A'):.2f}\"\n"
        content += f"echo -e \"Low:    {latest.get('low', 'N/A'):.2f}\"\n"
        content += f"echo -e \"Close:  {latest.get('close', 'N/A'):.2f}\"\n"
        content += f"echo -e \"Volume: {latest.get('volume', 'N/A'):.0f}\"\n"
        content += "echo\n"

        # Add key indicators section
        content += "echo -e \"$BOLD$YELLOW KEY INDICATORS $RESET\"\n"
        content += "echo -e \"$YELLOW -------------$RESET\"\n"

        # Add RSI
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            if rsi < 30:
                rsi_signal = "Oversold"
                rsi_color = "$GREEN"
            elif rsi > 70:
                rsi_signal = "Overbought"
                rsi_color = "$RED"
            else:
                rsi_signal = "Neutral"
                rsi_color = "$WHITE"
            content += f"echo -e \"RSI (14): {rsi:.2f} - {rsi_color}{rsi_signal}$RESET\"\n"

        # Add EMA Crossover
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            if ema_12 > ema_26:
                ema_signal = "Bullish"
                ema_color = "$GREEN"
            else:
                ema_signal = "Bearish"
                ema_color = "$RED"
            content += f"echo -e \"EMA Crossover: {ema_color}{ema_signal}$RESET (EMA12: {ema_12:.2f}, EMA26: {ema_26:.2f})\"\n"

        # Add MACD
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            macd_hist = macd - macd_signal
            if macd > macd_signal:
                macd_trend = "Bullish"
                macd_color = "$GREEN"
            else:
                macd_trend = "Bearish"
                macd_color = "$RED"
            content += f"echo -e \"MACD: {macd:.2f} - {macd_color}{macd_trend}$RESET (Signal: {macd_signal:.2f}, Hist: {macd_hist:.2f})\"\n"

        # Add Bollinger Bands
        if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_lower = latest['BBL_20_2.0']
            bb_middle = latest['BBM_20_2.0']
            bb_upper = latest['BBU_20_2.0']
            close = latest['close']
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50

            if close < bb_lower:
                bb_signal = "Lower Band"
                bb_color = "$GREEN"
            elif close > bb_upper:
                bb_signal = "Upper Band"
                bb_color = "$RED"
            else:
                bb_signal = "Middle Band"
                bb_color = "$WHITE"

            content += f"echo -e \"Bollinger Bands: {bb_color}{bb_signal}$RESET (Position: {bb_position:.1f}%)\"\n"

        content += "echo\n"

        # Add overall signal if available
        if 'signal' in df.columns:
            signal_value = latest['signal']

            if signal_value > 0.6:
                signal_text = "Strong Buy"
                signal_color = "$BOLD$GREEN"
            elif signal_value > 0:
                signal_text = "Buy"
                signal_color = "$GREEN"
            elif signal_value < -0.6:
                signal_text = "Strong Sell"
                signal_color = "$BOLD$RED"
            elif signal_value < 0:
                signal_text = "Sell"
                signal_color = "$RED"
            else:
                signal_text = "Neutral"
                signal_color = "$WHITE"

            content += f"echo -e \"$BOLD OVERALL SIGNAL: {signal_color}{signal_text}$RESET\"\n"
            content += "echo\n"

        # Add prediction if available
        if 'prediction' in df.columns:
            pred_value = latest['prediction']
            pred_prob = latest.get('prediction_probability', 0.5)

            if pred_value == 1:
                pred_text = "Bullish"
                pred_color = "$GREEN"
            else:
                pred_text = "Bearish"
                pred_color = "$RED"

            content += f"echo -e \"$BOLD PREDICTION: {pred_color}{pred_text}$RESET (Confidence: {pred_prob:.2f})\"\n"
            content += "echo\n"

        # Add recent signals section
        if 'signal' in df.columns:
            content += "echo -e \"$BOLD$YELLOW RECENT SIGNALS $RESET\"\n"
            content += "echo -e \"$YELLOW --------------$RESET\"\n"
            content += "echo -e \"Date                    Signal    Close\"\n"
            content += "echo -e \"---------------------------------------\"\n"

            # Get recent signals (last 10 periods)
            recent_df = df.tail(10).copy()
            recent_df['date'] = recent_df.index

            # Add signals
            signal_count = 0
            for _, row in recent_df.iterrows():
                if row['signal'] == 1:
                    signal_text = "BUY"
                    signal_color = "$GREEN"
                elif row['signal'] == -1:
                    signal_text = "SELL"
                    signal_color = "$RED"
                else:
                    continue  # Skip neutral signals

                date_str = str(row['date'])
                price_str = f"{row['close']:.2f}"
                content += f"echo -e \"{date_str}    {signal_color}{signal_text}$RESET      {price_str}\"\n"
                signal_count += 1

            if signal_count == 0:
                content += "echo -e \"No recent signals found.\"\n"

            content += "echo\n"

        # Add performance metrics if available
        if performance_metrics or trading_metrics:
            content += "echo -e \"$BOLD$YELLOW PERFORMANCE METRICS $RESET\"\n"
            content += "echo -e \"$YELLOW -------------------$RESET\"\n"

            if performance_metrics:
                for key, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    content += f"echo -e \"{key.replace('_', ' ').title()}: {formatted_value}\"\n"

            if trading_metrics:
                content += "echo\n"
                content += "echo -e \"$BOLD$YELLOW TRADING METRICS $RESET\"\n"
                content += "echo -e \"$YELLOW --------------$RESET\"\n"
                for key, value in trading_metrics.items():
                    if isinstance(value, (int, float)):
                        if key in ['strategy_return', 'buy_hold_return', 'annualized_strategy_return', 'annualized_buy_hold_return']:
                            if value > 0:
                                color = "$GREEN"
                            elif value < 0:
                                color = "$RED"
                            else:
                                color = "$WHITE"
                            formatted_value = f"{color}{value*100:.2f}%$RESET" if not pd.isna(value) else "N/A"
                        elif key in ['win_rate']:
                            formatted_value = f"{value*100:.2f}%" if not pd.isna(value) else "N/A"
                        else:
                            formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    content += f"echo -e \"{key.replace('_', ' ').title()}: {formatted_value}\"\n"

            content += "echo\n"

        # Add footer
        content += "echo -e \"Press Enter to close this window...\"\n"
        content += "read\n"

        return content

    def open_terminal(self, script_path):
        """Open a new terminal window and run the script."""
        if not script_path or not os.path.exists(script_path):
            logger.error(f"Script file not found: {script_path}")
            return False

        try:
            if self.system == 'Windows':
                # On Windows, use start command to open a new terminal window
                subprocess.Popen(['start', 'cmd', '/c', script_path], shell=True)
            elif self.system == 'Darwin':  # macOS
                # On macOS, use open command with Terminal.app
                subprocess.Popen(['open', '-a', 'Terminal', script_path])
            else:  # Linux
                # Try to detect the terminal emulator
                terminals = ['gnome-terminal', 'xterm', 'konsole', 'terminator']
                terminal_found = False

                for terminal in terminals:
                    try:
                        if terminal == 'gnome-terminal':
                            subprocess.Popen([terminal, '--', 'bash', script_path])
                        else:
                            subprocess.Popen([terminal, '-e', f'bash {script_path}'])
                        terminal_found = True
                        break
                    except FileNotFoundError:
                        continue

                if not terminal_found:
                    logger.error("Could not find a suitable terminal emulator")
                    return False

            logger.info(f"Opened terminal with script: {script_path}")
            return True

        except Exception as e:
            logger.error(f"Error opening terminal: {e}")
            return False
