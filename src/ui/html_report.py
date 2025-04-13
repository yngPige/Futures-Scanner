"""
HTML Report Generator for Crypto Futures Scanner

This module generates HTML reports for analysis results.
"""

import os
import pandas as pd
import numpy as np
import logging
import webbrowser
from datetime import datetime

logger = logging.getLogger(__name__)

class HTMLReportGenerator:
    """Generate HTML reports for analysis results."""
    
    def __init__(self, theme='dark'):
        """Initialize the HTML report generator."""
        self.theme = theme
        self.reports_dir = 'reports'
        
        # Create reports directory if it doesn't exist
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def generate_report(self, df, symbol, timeframe, report_type='analysis', 
                        performance_metrics=None, trading_metrics=None):
        """
        Generate an HTML report for the given data.
        
        Args:
            df (pd.DataFrame): DataFrame with analysis data
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the analysis
            report_type (str): Type of report ('analysis', 'prediction', 'backtest', 'all')
            performance_metrics (dict, optional): Performance metrics from backtest
            trading_metrics (dict, optional): Trading metrics from backtest
            
        Returns:
            str: Path to the generated HTML report
        """
        try:
            # Create report filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{report_type}_{timestamp}.html"
            report_path = os.path.join(self.reports_dir, filename)
            
            # Generate HTML content
            html_content = self._generate_html_content(df, symbol, timeframe, report_type, 
                                                      performance_metrics, trading_metrics)
            
            # Write HTML to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated: {report_path}")
            return report_path
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _generate_html_content(self, df, symbol, timeframe, report_type, 
                              performance_metrics, trading_metrics):
        """Generate the HTML content for the report."""
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Set theme colors
        if self.theme == 'dark':
            bg_color = '#1E1E1E'
            text_color = '#FFFFFF'
            card_bg = '#2D2D2D'
            border_color = '#3D3D3D'
            header_bg = '#0D47A1'
            positive_color = '#4CAF50'
            negative_color = '#F44336'
            neutral_color = '#9E9E9E'
        else:
            bg_color = '#F5F5F5'
            text_color = '#212121'
            card_bg = '#FFFFFF'
            border_color = '#E0E0E0'
            header_bg = '#2196F3'
            positive_color = '#4CAF50'
            negative_color = '#F44336'
            neutral_color = '#9E9E9E'
        
        # Start HTML content
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_type.title()} Report - {symbol} ({timeframe})</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: {header_bg};
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.8;
        }}
        .card {{
            background-color: {card_bg};
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            border-bottom: 1px solid {border_color};
            padding-bottom: 10px;
            font-size: 18px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metric {{
            padding: 15px;
            border-radius: 5px;
            background-color: rgba(255,255,255,0.05);
            border: 1px solid {border_color};
        }}
        .metric-name {{
            font-size: 14px;
            opacity: 0.8;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
        }}
        .positive {{
            color: {positive_color};
        }}
        .negative {{
            color: {negative_color};
        }}
        .neutral {{
            color: {neutral_color};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid {border_color};
        }}
        th {{
            background-color: rgba(0,0,0,0.1);
            font-weight: 600;
        }}
        tr:hover {{
            background-color: rgba(255,255,255,0.05);
        }}
        .tab {{
            overflow: hidden;
            border: 1px solid {border_color};
            background-color: {card_bg};
            border-radius: 5px 5px 0 0;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            color: {text_color};
            font-size: 16px;
        }}
        .tab button:hover {{
            background-color: rgba(255,255,255,0.1);
        }}
        .tab button.active {{
            background-color: {header_bg};
            color: white;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid {border_color};
            border-top: none;
            border-radius: 0 0 5px 5px;
            background-color: {card_bg};
            animation: fadeEffect 1s;
        }}
        @keyframes fadeEffect {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report_type.title()} Report - {symbol} ({timeframe})</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'Summary')">Summary</button>
            <button class="tablinks" onclick="openTab(event, 'Indicators')">Indicators</button>
            <button class="tablinks" onclick="openTab(event, 'Signals')">Signals</button>
            <button class="tablinks" onclick="openTab(event, 'Predictions')">Predictions</button>
            <button class="tablinks" onclick="openTab(event, 'Performance')">Performance</button>
        </div>
"""
        
        # Summary Tab
        html += f"""
        <div id="Summary" class="tabcontent" style="display: block;">
            <div class="card">
                <h2>Market Overview</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-name">Open</div>
                        <div class="metric-value">{latest.get('open', 'N/A'):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">High</div>
                        <div class="metric-value">{latest.get('high', 'N/A'):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Low</div>
                        <div class="metric-value">{latest.get('low', 'N/A'):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Close</div>
                        <div class="metric-value">{latest.get('close', 'N/A'):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Volume</div>
                        <div class="metric-value">{latest.get('volume', 'N/A'):.0f}</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Key Indicators</h2>
                <div class="grid">
"""
        
        # Add key indicators
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            rsi_class = "positive" if rsi < 30 else "negative" if rsi > 70 else "neutral"
            rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">RSI (14)</div>
                        <div class="metric-value {rsi_class}">{rsi:.2f} - {rsi_signal}</div>
                    </div>
"""
        
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            ema_12 = latest['ema_12']
            ema_26 = latest['ema_26']
            ema_class = "positive" if ema_12 > ema_26 else "negative"
            ema_signal = "Bullish" if ema_12 > ema_26 else "Bearish"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">EMA Crossover</div>
                        <div class="metric-value {ema_class}">{ema_signal}</div>
                    </div>
"""
        
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            macd_hist = macd - macd_signal
            macd_class = "positive" if macd > macd_signal else "negative"
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">MACD</div>
                        <div class="metric-value {macd_class}">{macd:.2f} - {macd_trend}</div>
                    </div>
"""
        
        if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_lower = latest['BBL_20_2.0']
            bb_middle = latest['BBM_20_2.0']
            bb_upper = latest['BBU_20_2.0']
            close = latest['close']
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
            bb_class = "positive" if close < bb_lower else "negative" if close > bb_upper else "neutral"
            bb_signal = "Lower Band" if close < bb_lower else "Upper Band" if close > bb_upper else "Middle Band"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">Bollinger Bands</div>
                        <div class="metric-value {bb_class}">{bb_signal} ({bb_position:.1f}%)</div>
                    </div>
"""
        
        # Add overall signal if available
        if 'signal' in df.columns:
            signal_value = latest['signal']
            signal_class = "positive" if signal_value > 0 else "negative" if signal_value < 0 else "neutral"
            signal_text = "Strong Buy" if signal_value > 0.6 else \
                         "Buy" if signal_value > 0 else \
                         "Strong Sell" if signal_value < -0.6 else \
                         "Sell" if signal_value < 0 else "Neutral"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">Overall Signal</div>
                        <div class="metric-value {signal_class}">{signal_text}</div>
                    </div>
"""
        
        # Add prediction if available
        if 'prediction' in df.columns:
            pred_value = latest['prediction']
            pred_prob = latest.get('prediction_probability', 0.5)
            pred_class = "positive" if pred_value == 1 else "negative"
            pred_text = "Bullish" if pred_value == 1 else "Bearish"
            html += f"""
                    <div class="metric">
                        <div class="metric-name">Prediction</div>
                        <div class="metric-value {pred_class}">{pred_text} ({pred_prob:.2f})</div>
                    </div>
"""
        
        html += """
                </div>
            </div>
        </div>
"""
        
        # Indicators Tab
        html += """
        <div id="Indicators" class="tabcontent">
            <div class="card">
                <h2>Technical Indicators</h2>
                <table>
                    <tr>
                        <th>Indicator</th>
                        <th>Value</th>
                        <th>Signal</th>
                    </tr>
"""
        
        # Add indicators to the table
        indicator_groups = {
            "Trend": ["ema_12", "ema_26", "sma_20", "sma_50", "sma_200", "adx_14"],
            "Momentum": ["rsi_14", "stoch_14_3_3", "stochd_14_3_3", "cci_20", "roc_10"],
            "Volatility": ["atr_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"],
            "Volume": ["volume", "obv", "cmf_20"],
            "Oscillators": ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]
        }
        
        for group, indicators in indicator_groups.items():
            html += f"""
                    <tr>
                        <td colspan="3" style="background-color: rgba(0,0,0,0.2); font-weight: bold;">{group}</td>
                    </tr>
"""
            
            for indicator in indicators:
                if indicator in df.columns:
                    value = latest[indicator]
                    
                    # Determine signal based on indicator
                    signal = "Neutral"
                    signal_class = "neutral"
                    if indicator == "rsi_14":
                        signal = "Oversold" if value < 30 else "Overbought" if value > 70 else "Neutral"
                        signal_class = "positive" if value < 30 else "negative" if value > 70 else "neutral"
                    elif indicator in ["ema_12", "sma_20"] and "ema_26" in df.columns:
                        signal = "Bullish" if value > latest["ema_26"] else "Bearish"
                        signal_class = "positive" if value > latest["ema_26"] else "negative"
                    elif indicator == "adx_14":
                        signal = "Strong Trend" if value > 25 else "Weak Trend"
                        signal_class = "positive" if value > 25 else "neutral"
                    elif indicator == "MACD_12_26_9" and "MACDs_12_26_9" in df.columns:
                        signal = "Bullish" if value > latest["MACDs_12_26_9"] else "Bearish"
                        signal_class = "positive" if value > latest["MACDs_12_26_9"] else "negative"
                    elif indicator == "BBP_20_2.0":
                        signal = "Oversold" if value < 0 else "Overbought" if value > 1 else "Neutral"
                        signal_class = "positive" if value < 0 else "negative" if value > 1 else "neutral"
                    
                    # Format the value
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"""
                    <tr>
                        <td>{indicator}</td>
                        <td>{formatted_value}</td>
                        <td class="{signal_class}">{signal}</td>
                    </tr>
"""
        
        html += """
                </table>
            </div>
        </div>
"""
        
        # Signals Tab
        html += """
        <div id="Signals" class="tabcontent">
            <div class="card">
                <h2>Recent Trading Signals</h2>
"""
        
        # Get signals from the dataframe
        if 'signal' not in df.columns:
            html += "<p>No trading signals available.</p>"
        else:
            # Get recent signals (last 20 periods)
            recent_df = df.tail(20).copy()
            recent_df['date'] = recent_df.index
            
            # Add signals table
            html += """
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Signal</th>
                        <th>Close Price</th>
                    </tr>
"""
            
            # Add signals
            for _, row in recent_df.iterrows():
                if row['signal'] == 1:
                    signal_text = "BUY"
                    signal_class = "positive"
                elif row['signal'] == -1:
                    signal_text = "SELL"
                    signal_class = "negative"
                else:
                    continue  # Skip neutral signals
                
                date_str = str(row['date'])
                price_str = f"{row['close']:.2f}"
                html += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td class="{signal_class}">{signal_text}</td>
                        <td>{price_str}</td>
                    </tr>
"""
            
            html += """
                </table>
"""
            
            # Add signal distribution
            buy_signals = len(df[df['signal'] == 1])
            sell_signals = len(df[df['signal'] == -1])
            neutral_signals = len(df[df['signal'] == 0])
            total_signals = buy_signals + sell_signals + neutral_signals
            
            html += f"""
                <h2>Signal Distribution</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-name">Buy Signals</div>
                        <div class="metric-value positive">{buy_signals} ({buy_signals/total_signals*100:.1f}%)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Sell Signals</div>
                        <div class="metric-value negative">{sell_signals} ({sell_signals/total_signals*100:.1f}%)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Neutral Signals</div>
                        <div class="metric-value neutral">{neutral_signals} ({neutral_signals/total_signals*100:.1f}%)</div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Predictions Tab
        html += """
        <div id="Predictions" class="tabcontent">
            <div class="card">
                <h2>Recent Predictions</h2>
"""
        
        # Check if predictions are available
        if 'prediction' not in df.columns:
            html += "<p>No predictions available.</p>"
        else:
            # Get recent predictions (last 20 periods)
            recent_df = df.tail(20).copy()
            recent_df['date'] = recent_df.index
            
            # Add predictions table
            html += """
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Prediction</th>
                        <th>Probability</th>
                        <th>Close Price</th>
                    </tr>
"""
            
            # Add predictions
            for _, row in recent_df.iterrows():
                date_str = str(row['date'])
                pred_value = row['prediction']
                pred_text = "BULLISH" if pred_value == 1 else "BEARISH"
                pred_class = "positive" if pred_value == 1 else "negative"
                prob_value = row.get('prediction_probability', 0.5)
                prob_str = f"{prob_value:.2f}"
                price_str = f"{row['close']:.2f}"
                
                html += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td class="{pred_class}">{pred_text}</td>
                        <td>{prob_str}</td>
                        <td>{price_str}</td>
                    </tr>
"""
            
            html += """
                </table>
"""
            
            # Add prediction distribution
            bullish_preds = len(df[df['prediction'] == 1])
            bearish_preds = len(df[df['prediction'] == 0])
            total_preds = bullish_preds + bearish_preds
            
            html += f"""
                <h2>Prediction Distribution</h2>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-name">Bullish Predictions</div>
                        <div class="metric-value positive">{bullish_preds} ({bullish_preds/total_preds*100:.1f}%)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-name">Bearish Predictions</div>
                        <div class="metric-value negative">{bearish_preds} ({bearish_preds/total_preds*100:.1f}%)</div>
                    </div>
                </div>
"""
            
            # Add prediction accuracy if target is available
            if 'target' in df.columns:
                correct_preds = len(df[df['prediction'] == df['target']])
                accuracy = correct_preds / total_preds * 100
                html += f"""
                <div class="metric" style="margin-top: 15px;">
                    <div class="metric-name">Prediction Accuracy</div>
                    <div class="metric-value">{accuracy:.2f}%</div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Performance Tab
        html += """
        <div id="Performance" class="tabcontent">
            <div class="card">
                <h2>Backtest Performance</h2>
"""
        
        if not performance_metrics and not trading_metrics:
            html += "<p>No performance metrics available.</p>"
        else:
            # Add performance metrics
            if performance_metrics:
                html += """
                <h3>Prediction Performance</h3>
                <div class="grid">
"""
                for key, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"""
                    <div class="metric">
                        <div class="metric-name">{key.replace('_', ' ').title()}</div>
                        <div class="metric-value">{formatted_value}</div>
                    </div>
"""
                html += """
                </div>
"""
            
            # Add trading metrics
            if trading_metrics:
                html += """
                <h3>Trading Performance</h3>
                <div class="grid">
"""
                for key, value in trading_metrics.items():
                    if isinstance(value, (int, float)):
                        if key in ['strategy_return', 'buy_hold_return', 'annualized_strategy_return', 'annualized_buy_hold_return']:
                            formatted_value = f"{value*100:.2f}%" if not pd.isna(value) else "N/A"
                            metric_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                        elif key in ['win_rate']:
                            formatted_value = f"{value*100:.2f}%" if not pd.isna(value) else "N/A"
                            metric_class = "neutral"
                        else:
                            formatted_value = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
                            metric_class = "neutral"
                    else:
                        formatted_value = str(value)
                        metric_class = "neutral"
                    
                    html += f"""
                    <div class="metric">
                        <div class="metric-name">{key.replace('_', ' ').title()}</div>
                        <div class="metric-value {metric_class}">{formatted_value}</div>
                    </div>
"""
                html += """
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # End HTML content
        html += """
    </div>
    
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""
        
        return html
    
    def open_report(self, report_path):
        """Open the report in the default web browser."""
        if report_path and os.path.exists(report_path):
            webbrowser.open('file://' + os.path.abspath(report_path))
            return True
        return False
