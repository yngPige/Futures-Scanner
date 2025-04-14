"""
Test script for timeframe display, prediction, and trading levels on charts.

This script creates a sample chart with timeframe display, prediction, and entry/exit levels.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create sample data
def create_sample_data(n_points=100):
    """Create sample OHLCV data with predictions."""
    today = datetime.now()
    dates = [today - timedelta(hours=n_points-i) for i in range(n_points)]

    # Generate price data with some randomness
    base_price = 70000  # Base price for BTC
    prices = []
    for i in range(n_points):
        if i == 0:
            prices.append(base_price)
        else:
            # Random walk with some mean reversion
            change = np.random.normal(0, 500) + (base_price - prices[-1]) * 0.05
            prices.append(prices[-1] + change)

    # Create OHLCV data
    data = {
        'open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(n_points)]
    }

    # Add prediction data
    # Generate random predictions (1 for bullish, 0 for bearish)
    data['prediction'] = [np.random.choice([0, 1]) for _ in range(n_points)]

    # Generate random prediction probabilities
    data['prediction_probability'] = [np.random.uniform(0.5, 0.95) for _ in range(n_points)]

    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    return df

def create_chart_with_timeframe(df, timeframe='1h'):
    """Create a chart with timeframe display and prediction."""
    # Create figure
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3],
                       subplot_titles=('BTC/USDT', 'Volume'))

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

    # Add volume chart
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker=dict(color=colors)
        ),
        row=2, col=1
    )

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
        if pred_text == "BULLISH":
            entry_price = latest_price * 0.995  # Slightly below current price
            stop_loss = entry_price * 0.97     # 3% below entry
            take_profit = entry_price * 1.05   # 5% above entry
        else:  # BEARISH
            entry_price = latest_price * 1.005  # Slightly above current price
            stop_loss = entry_price * 1.03     # 3% above entry
            take_profit = entry_price * 0.95   # 5% below entry

        # Add prediction annotation in the upper left corner
        fig.add_annotation(
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>PREDICTION: {pred_text}</b> ({pred_prob:.2f})",
            showarrow=False,
            font=dict(
                size=16,
                color=pred_color
            ),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=pred_color,
            borderwidth=2,
            borderpad=4,
            align='left',
            xanchor='left',
            yanchor='top'
        )

        # Add entry price line
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[entry_price, entry_price],
                name='Entry Price',
                line=dict(color='yellow', width=2, dash='dash'),
                opacity=0.7
            )
        )

        # Add entry price annotation
        fig.add_annotation(
            x=df.index[-1],
            y=entry_price,
            text=f"Entry: {entry_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='yellow',
            arrowsize=1,
            arrowwidth=2,
            ax=70,
            ay=0,
            font=dict(color='yellow', size=12)
        )

        # Add stop loss line
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[stop_loss, stop_loss],
                name='Stop Loss',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            )
        )

        # Add stop loss annotation
        fig.add_annotation(
            x=df.index[-1],
            y=stop_loss,
            text=f"Stop Loss: {stop_loss:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            arrowsize=1,
            arrowwidth=2,
            ax=70,
            ay=0,
            font=dict(color='red', size=12)
        )

        # Add take profit line
        fig.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[take_profit, take_profit],
                name='Take Profit',
                line=dict(color='green', width=2, dash='dash'),
                opacity=0.7
            )
        )

        # Add take profit annotation
        fig.add_annotation(
            x=df.index[-1],
            y=take_profit,
            text=f"Take Profit: {take_profit:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            arrowsize=1,
            arrowwidth=2,
            ax=70,
            ay=0,
            font=dict(color='green', size=12)
        )

        # Calculate risk-reward ratio
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        if risk_amount > 0:
            risk_reward_ratio = reward_amount / risk_amount

            # Add risk-reward ratio annotation
            fig.add_annotation(
                x=0,
                y=0.97,
                xref="paper",
                yref="paper",
                text=f"Risk/Reward: 1:{risk_reward_ratio:.2f}",
                showarrow=False,
                font=dict(
                    size=14,
                    color='white'
                ),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='gray',
                borderwidth=1,
                borderpad=4,
                align='left',
                xanchor='left',
                yanchor='top'
            )

    # Format current date and time
    current_time = datetime.now().strftime("%b %d, %Y %H:%M UTC")

    # Add timeframe annotation in the upper right corner
    fig.add_annotation(
        x=1,
        y=1,
        xref="paper",
        yref="paper",
        text=f"<b>{timeframe}</b>",
        showarrow=False,
        font=dict(
            size=16,
            color='white'
        ),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4,
        align='right',
        xanchor='right',
        yanchor='top'
    )

    # Add timestamp below the timeframe
    fig.add_annotation(
        x=1,
        y=0.97,
        xref="paper",
        yref="paper",
        text=f"<i>{current_time}</i>",
        showarrow=False,
        font=dict(
            size=12,
            color='white'
        ),
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4,
        align='right',
        xanchor='right',
        yanchor='top'
    )

    # Add 3lacks branding
    fig.add_annotation(
        x=1,
        y=0.94,
        xref="paper",
        yref="paper",
        text="3lack_Hands",
        showarrow=False,
        font=dict(
            size=12,
            color='white'
        ),
        bgcolor='rgba(0,0,0,0.0)',
        align='right',
        xanchor='right',
        yanchor='top'
    )

    # Update layout
    fig.update_layout(
        title='BTC/USDT Chart with Timeframe Display',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800,
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
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

def main():
    """Main function."""
    print("Testing timeframe display on charts")

    # Create sample data
    df = create_sample_data()

    # Create chart with timeframe display
    fig = create_chart_with_timeframe(df, timeframe='1h')

    # Show chart
    fig.show()

    print("Test completed successfully")

if __name__ == "__main__":
    main()
