"""
Chart Generator Module for Crypto Futures Scanner

This module provides functionality to create charts and visualizations
for cryptocurrency data and technical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChartGenerator:
    """Class to generate charts for cryptocurrency data and technical analysis."""

    def __init__(self, theme='dark'):
        """
        Initialize the ChartGenerator.

        Args:
            theme (str): Chart theme ('dark' or 'light')
        """
        self.theme = theme
        logger.info(f"Initialized ChartGenerator with {theme} theme")

    def create_candlestick_chart(self, df, title='Price Chart with Indicators', save_path=None):
        """
        Create a candlestick chart using mplfinance.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            title (str): Chart title
            save_path (str, optional): Path to save the chart

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create chart")
            return None

        try:
            # Set theme
            if self.theme == 'dark':
                mpf_style = 'nightclouds'
            else:
                mpf_style = 'yahoo'

            # Standardize column names to lowercase
            df_cols = [col.lower() if isinstance(col, str) else
                      (col[0].lower() if isinstance(col, tuple) else col)
                      for col in df.columns]

            # Check if we have OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            has_ohlc = all(col in df_cols for col in required_cols)

            if not has_ohlc:
                logger.error(f"Missing required OHLC columns. Found: {df.columns}")
                return None

            # Create a copy of the DataFrame with standardized column names for mplfinance
            ohlc_df = df.copy()

            # Ensure column names are correct for mplfinance
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df_cols and new_col not in ohlc_df.columns:
                    # Find the actual column name (might be capitalized differently)
                    actual_col = [col for col in df.columns if col.lower() == old_col][0]
                    ohlc_df[new_col] = ohlc_df[actual_col]

            # Make sure the index is a DatetimeIndex
            if not isinstance(ohlc_df.index, pd.DatetimeIndex):
                logger.warning("Converting index to DatetimeIndex for mplfinance")
                try:
                    ohlc_df.index = pd.to_datetime(ohlc_df.index)
                except Exception as e:
                    logger.error(f"Could not convert index to DatetimeIndex: {e}")
                    return None

            # Create a copy with just the OHLCV columns for mplfinance
            ohlc_plot_df = ohlc_df[['Open', 'High', 'Low', 'Close']].copy()
            if 'Volume' in ohlc_df.columns:
                ohlc_plot_df['Volume'] = ohlc_df['Volume']

            # Add buy/sell signals as markers if available
            markers = []
            if 'signal' in df.columns:
                buy_signals = df[df['signal'] == 1]
                sell_signals = df[df['signal'] == -1]

                # If there are too many signals, sample them to reduce clutter
                max_signals = 10  # Maximum number of signals to display
                if len(buy_signals) > max_signals:
                    buy_signals = buy_signals.sample(max_signals, random_state=42)
                if len(sell_signals) > max_signals:
                    sell_signals = sell_signals.sample(max_signals, random_state=42)

                # Add buy signals
                for idx in buy_signals.index:
                    if idx in ohlc_plot_df.index:
                        markers.append(mpf.make_addplot(
                            pd.Series(ohlc_plot_df.loc[idx, 'Low'] * 0.99, index=[idx]),
                            type='scatter', marker='^', markersize=100, color='green'
                        ))

                # Add sell signals
                for idx in sell_signals.index:
                    if idx in ohlc_plot_df.index:
                        markers.append(mpf.make_addplot(
                            pd.Series(ohlc_plot_df.loc[idx, 'High'] * 1.01, index=[idx]),
                            type='scatter', marker='v', markersize=100, color='red'
                        ))

            # Add predictions as markers if available
            if 'prediction' in df.columns:
                if 'prediction_probability' in df.columns:
                    # Only show high-confidence predictions (probability > 0.7)
                    buy_preds = df[(df['prediction'] == 1) & (df['prediction_probability'] > 0.7)]
                    sell_preds = df[(df['prediction'] == 0) & (df['prediction_probability'] > 0.7)]
                else:
                    # If probability is not available, just sample a few predictions
                    buy_preds = df[df['prediction'] == 1]
                    sell_preds = df[df['prediction'] == 0]

                # If there are too many predictions, sample them to reduce clutter
                max_preds = 15  # Maximum number of predictions to display
                if len(buy_preds) > max_preds:
                    buy_preds = buy_preds.sample(max_preds, random_state=42)
                if len(sell_preds) > max_preds:
                    sell_preds = sell_preds.sample(max_preds, random_state=42)

                # Add bullish predictions
                for idx in buy_preds.index:
                    if idx in ohlc_plot_df.index:
                        markers.append(mpf.make_addplot(
                            pd.Series(ohlc_plot_df.loc[idx, 'Low'] * 0.98, index=[idx]),
                            type='scatter', marker='o', markersize=50, color='lime'
                        ))

                # Add bearish predictions
                for idx in sell_preds.index:
                    if idx in ohlc_plot_df.index:
                        markers.append(mpf.make_addplot(
                            pd.Series(ohlc_plot_df.loc[idx, 'High'] * 1.02, index=[idx]),
                            type='scatter', marker='o', markersize=50, color='orange'
                        ))

            # Add technical indicators as plots
            if 'rsi_14' in df.columns:
                markers.append(mpf.make_addplot(
                    df['rsi_14'], panel=1, color='purple', secondary_y=False
                ))

            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                markers.append(mpf.make_addplot(
                    df['MACD_12_26_9'], panel=2, color='blue', secondary_y=False
                ))
                markers.append(mpf.make_addplot(
                    df['MACDs_12_26_9'], panel=2, color='red', secondary_y=False
                ))

            # Plot the candlestick chart with all markers
            fig, axes = mpf.plot(
                ohlc_plot_df,
                type='candle',
                style=mpf_style,
                title=title,
                figsize=(12, 10),
                volume=True if 'Volume' in ohlc_plot_df.columns else False,
                panel_ratios=(4, 1, 1) if 'MACD_12_26_9' in df.columns else (4, 1),
                addplot=markers if markers else None,
                returnfig=True
            )

            # Save the figure if a path is provided
            if save_path:
                fig.savefig(save_path)
                logger.info(f"Chart saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_price_with_indicators(self, df, title='Price Chart with Indicators', save_path=None):
        """
        Create a price chart with key indicators using Matplotlib or mplfinance.

        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            title (str): Chart title
            save_path (str, optional): Path to save the chart

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Try to create a candlestick chart first
        fig = self.create_candlestick_chart(df, title, save_path)
        if fig is not None:
            return fig

        # If candlestick chart fails, fall back to the original line chart
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create chart")
            return None

        try:
            # Create figure and axes
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

            # Set theme
            if self.theme == 'dark':
                plt.style.use('dark_background')

            # Standardize column names to lowercase
            df_cols = [col.lower() if isinstance(col, str) else
                      (col[0].lower() if isinstance(col, tuple) else col)
                      for col in df.columns]

            # Create a mapping from lowercase to actual column names
            col_mapping = {col.lower() if isinstance(col, str) else col: col for col in df.columns}

            # Plot price as a line
            if 'close' in df_cols:
                close_col = col_mapping.get('close', 'Close')
                ax1.plot(df.index, df[close_col], label='Close Price', linewidth=2)
            elif 'Close' in df.columns:
                ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
            else:
                logger.error(f"Could not find 'close' column in {df.columns}")
                return None

            # Plot moving averages if available
            if 'sma_20' in df.columns:
                ax1.plot(df.index, df['sma_20'], label='SMA 20', linewidth=1.5, alpha=0.8)
            if 'sma_50' in df.columns:
                ax1.plot(df.index, df['sma_50'], label='SMA 50', linewidth=1.5, alpha=0.8)
            if 'ema_12' in df.columns:
                ax1.plot(df.index, df['ema_12'], label='EMA 12', linewidth=1.5, alpha=0.8)
            if 'ema_26' in df.columns:
                ax1.plot(df.index, df['ema_26'], label='EMA 26', linewidth=1.5, alpha=0.8)

            # Plot Bollinger Bands if available
            if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                ax1.plot(df.index, df['BBU_20_2.0'], 'g--', label='Upper BB', alpha=0.5)
                ax1.plot(df.index, df['BBM_20_2.0'], 'g-', label='Middle BB', alpha=0.5)
                ax1.plot(df.index, df['BBL_20_2.0'], 'g--', label='Lower BB', alpha=0.5)
                ax1.fill_between(df.index, df['BBL_20_2.0'], df['BBU_20_2.0'], alpha=0.1, color='green')

            # Plot buy/sell signals if available - with reduced frequency
            if 'signal' in df.columns:
                # Filter signals to reduce clutter
                buy_signals = df[df['signal'] == 1]
                sell_signals = df[df['signal'] == -1]

                # If there are too many signals, sample them to reduce clutter
                max_signals = 10  # Maximum number of signals to display
                if len(buy_signals) > max_signals:
                    buy_signals = buy_signals.sample(max_signals, random_state=42)
                if len(sell_signals) > max_signals:
                    sell_signals = sell_signals.sample(max_signals, random_state=42)

                ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=120, label='Buy Signal')
                ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=120, label='Sell Signal')

            # Plot predictions if available - with reduced frequency
            if 'prediction' in df.columns:
                if 'prediction_probability' in df.columns:
                    # Only show high-confidence predictions (probability > 0.7)
                    buy_preds = df[(df['prediction'] == 1) & (df['prediction_probability'] > 0.7)]
                    sell_preds = df[(df['prediction'] == 0) & (df['prediction_probability'] > 0.7)]
                else:
                    # If probability is not available, just sample a few predictions
                    buy_preds = df[df['prediction'] == 1]
                    sell_preds = df[df['prediction'] == 0]

                # If there are too many predictions, sample them to reduce clutter
                max_preds = 15  # Maximum number of predictions to display
                if len(buy_preds) > max_preds:
                    buy_preds = buy_preds.sample(max_preds, random_state=42)
                if len(sell_preds) > max_preds:
                    sell_preds = sell_preds.sample(max_preds, random_state=42)

                ax1.scatter(buy_preds.index, buy_preds['close'], marker='o', color='lime', s=60, label='Bullish Prediction')
                ax1.scatter(sell_preds.index, sell_preds['close'], marker='o', color='orange', s=60, label='Bearish Prediction')

            # Plot RSI in the second subplot if available
            if 'rsi_14' in df.columns:
                ax2.plot(df.index, df['rsi_14'], label='RSI (14)', color='purple')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('RSI')
                ax2.legend(loc='upper left')
            else:
                # Just add a placeholder
                ax2.set_ylabel('RSI (not available)')
                ax2.set_yticks([])

            # Plot MACD in the third subplot if available
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                ax3.plot(df.index, df['MACD_12_26_9'], label='MACD', color='blue')
                ax3.plot(df.index, df['MACDs_12_26_9'], label='Signal', color='red')

                # Plot histogram
                if 'MACDh_12_26_9' in df.columns:
                    ax3.bar(df.index, df['MACDh_12_26_9'], label='Histogram', color='green', alpha=0.5)

                ax3.set_ylabel('MACD')
                ax3.legend(loc='upper left')
            else:
                # Just add a placeholder
                ax3.set_ylabel('MACD (not available)')
                ax3.set_yticks([])

            # Format x-axis dates
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Set labels and title
            ax1.set_ylabel('Price')
            ax1.set_title(title)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Chart saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return None

    def create_interactive_chart(self, df, title='Interactive Price Chart', include_volume=True):
        """
        Create an interactive price chart with indicators using Plotly.

        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            title (str): Chart title
            include_volume (bool): Whether to include volume chart

        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot create interactive chart")
            return None

        try:
            # Determine number of rows based on available indicators
            rows = 2  # Price and volume

            if 'rsi_14' in df.columns:
                rows += 1

            if 'MACD_12_26_9' in df.columns:
                rows += 1

            # Create subplots
            fig = make_subplots(rows=rows, cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.03,
                               row_heights=[0.5] + [0.2] * (rows-1),
                               subplot_titles=(title, 'Volume', 'RSI', 'MACD')[:rows])

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # Add moving averages if available
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )

            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )

            if 'ema_12' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_12'],
                        name='EMA 12',
                        line=dict(color='purple', width=1)
                    ),
                    row=1, col=1
                )

            if 'ema_26' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_26'],
                        name='EMA 26',
                        line=dict(color='green', width=1)
                    ),
                    row=1, col=1
                )

            # Add Bollinger Bands if available
            if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BBU_20_2.0'],
                        name='Upper BB',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BBM_20_2.0'],
                        name='Middle BB',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BBL_20_2.0'],
                        name='Lower BB',
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.05)'
                    ),
                    row=1, col=1
                )

            # Add buy/sell signals if available - with reduced frequency
            if 'signal' in df.columns:
                # Filter signals to reduce clutter
                buy_signals = df[df['signal'] == 1]
                sell_signals = df[df['signal'] == -1]

                # If there are too many signals, sample them to reduce clutter
                max_signals = 10  # Maximum number of signals to display
                if len(buy_signals) > max_signals:
                    buy_signals = buy_signals.sample(max_signals, random_state=42)
                if len(sell_signals) > max_signals:
                    sell_signals = sell_signals.sample(max_signals, random_state=42)

                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        name='Buy Signal',
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=18,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        )
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        name='Sell Signal',
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=18,
                            color='red',
                            line=dict(width=1, color='darkred')
                        )
                    ),
                    row=1, col=1
                )

            # Add predictions if available - with reduced frequency
            if 'prediction' in df.columns:
                if 'prediction_probability' in df.columns:
                    # Only show high-confidence predictions (probability > 0.7)
                    buy_preds = df[(df['prediction'] == 1) & (df['prediction_probability'] > 0.7)]
                    sell_preds = df[(df['prediction'] == 0) & (df['prediction_probability'] > 0.7)]
                else:
                    # If probability is not available, just sample a few predictions
                    buy_preds = df[df['prediction'] == 1]
                    sell_preds = df[df['prediction'] == 0]

                # If there are too many predictions, sample them to reduce clutter
                max_preds = 15  # Maximum number of predictions to display
                if len(buy_preds) > max_preds:
                    buy_preds = buy_preds.sample(max_preds, random_state=42)
                if len(sell_preds) > max_preds:
                    sell_preds = sell_preds.sample(max_preds, random_state=42)

                fig.add_trace(
                    go.Scatter(
                        x=buy_preds.index,
                        y=buy_preds['close'],
                        name='Bullish Prediction',
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='lime',
                            line=dict(width=1, color='green')
                        )
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=sell_preds.index,
                        y=sell_preds['close'],
                        name='Bearish Prediction',
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='orange',
                            line=dict(width=1, color='red')
                        )
                    ),
                    row=1, col=1
                )

            # Add volume chart
            current_row = 2
            if include_volume and 'volume' in df.columns:
                colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]

                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker=dict(color=colors)
                    ),
                    row=current_row, col=1
                )
                current_row += 1

            # Add RSI if available
            if 'rsi_14' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['rsi_14'],
                        name='RSI (14)',
                        line=dict(color='purple', width=1)
                    ),
                    row=current_row, col=1
                )

                # Add overbought/oversold lines
                fig.add_shape(
                    type='line',
                    x0=df.index[0],
                    y0=70,
                    x1=df.index[-1],
                    y1=70,
                    line=dict(color='red', width=1, dash='dash'),
                    row=current_row, col=1
                )

                fig.add_shape(
                    type='line',
                    x0=df.index[0],
                    y0=30,
                    x1=df.index[-1],
                    y1=30,
                    line=dict(color='green', width=1, dash='dash'),
                    row=current_row, col=1
                )

                current_row += 1

            # Add MACD if available
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD_12_26_9'],
                        name='MACD',
                        line=dict(color='blue', width=1)
                    ),
                    row=current_row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACDs_12_26_9'],
                        name='Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=current_row, col=1
                )

                # Add histogram
                if 'MACDh_12_26_9' in df.columns:
                    colors = ['green' if val >= 0 else 'red' for val in df['MACDh_12_26_9']]

                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['MACDh_12_26_9'],
                            name='Histogram',
                            marker=dict(color=colors)
                        ),
                        row=current_row, col=1
                    )

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
                height=250 * rows,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)

            if include_volume:
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                if 'rsi_14' in df.columns:
                    fig.update_yaxes(title_text="RSI", row=3, col=1)

                    if 'MACD_12_26_9' in df.columns:
                        fig.update_yaxes(title_text="MACD", row=4, col=1)
            else:
                if 'rsi_14' in df.columns:
                    fig.update_yaxes(title_text="RSI", row=2, col=1)

                    if 'MACD_12_26_9' in df.columns:
                        fig.update_yaxes(title_text="MACD", row=3, col=1)

            logger.info("Created interactive chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating interactive chart: {e}")
            return None

    def plot_model_performance(self, actual, predicted, title='Model Performance', save_path=None):
        """
        Create a chart showing model performance.

        Args:
            actual (pd.Series): Actual values
            predicted (pd.Series): Predicted values
            title (str): Chart title
            save_path (str, optional): Path to save the chart

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Set theme
            if self.theme == 'dark':
                plt.style.use('dark_background')

            # Plot actual vs predicted
            ax.plot(actual.index, actual, label='Actual', linewidth=2)
            ax.plot(predicted.index, predicted, label='Predicted', linewidth=2, alpha=0.7)

            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Adjust layout
            plt.tight_layout()

            # Save figure if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Performance chart saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Import required modules for example
    import yfinance as yf
    from src.analysis.technical_analysis import TechnicalAnalyzer

    # Download sample data
    data = yf.download('BTC-USD', period='3mo', interval='1d')

    # Add technical indicators
    analyzer = TechnicalAnalyzer()
    data_with_indicators = analyzer.add_all_indicators(data)
    data_with_signals = analyzer.generate_signals(data_with_indicators)

    # Initialize chart generator
    chart_gen = ChartGenerator(theme='dark')

    # Create price chart
    fig = chart_gen.plot_price_with_indicators(data_with_signals, title='BTC/USD with Indicators')
    plt.show()

    # Create interactive chart
    interactive_fig = chart_gen.create_interactive_chart(data_with_signals, title='BTC/USD Interactive Chart')
    interactive_fig.show()
