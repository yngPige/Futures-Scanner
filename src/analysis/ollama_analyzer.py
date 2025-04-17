"""
Ollama Integration Module for Crypto Futures Scanner

This module provides functionality to analyze market data using Ollama-hosted LLMs.
"""

import os
import json
import logging
import requests
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class OllamaAnalyzer:
    """Class to analyze market data using Ollama-hosted LLMs."""

    def __init__(self, model_name="llama3", host="http://localhost:11434"):
        """
        Initialize the Ollama Analyzer.

        Args:
            model_name (str): Name of the Ollama model to use (default: "llama3")
            host (str): Ollama host URL (default: "http://localhost:11434")
        """
        self.model_name = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        logger.info(f"Initialized OllamaAnalyzer with model: {model_name}")

    def check_ollama_available(self):
        """
        Check if Ollama is available and the specified model is installed.

        Returns:
            bool: True if Ollama is available and the model is installed, False otherwise
        """
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                logger.error(f"Ollama server not available at {self.host}")
                return False

            # Check if the model is installed
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {', '.join(model_names)}")
                return False
                
            logger.info(f"Ollama is available with model {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

    def _prepare_market_data(self, df):
        """
        Prepare market data for analysis.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and indicators

        Returns:
            dict: Dictionary with market data for analysis
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided, cannot prepare market data")
                return {}

            # Get the latest data point and recent price history
            latest = df.iloc[-1]
            recent_df = df.tail(10).copy()
            
            # Format recent prices for display
            recent_prices = {
                "open": [float(price) for price in recent_df['open'].tolist()],
                "high": [float(price) for price in recent_df['high'].tolist()],
                "low": [float(price) for price in recent_df['low'].tolist()],
                "close": [float(price) for price in recent_df['close'].tolist()],
                "volume": [float(vol) for vol in recent_df['volume'].tolist()]
            }
            
            # Prepare key indicators
            indicators = {}
            
            # Add RSI if available
            if 'rsi_14' in df.columns:
                rsi_value = float(latest['rsi_14'])
                rsi_trend = "overbought" if rsi_value > 70 else "oversold" if rsi_value < 30 else "neutral"
                indicators["rsi"] = {
                    "value": rsi_value,
                    "interpretation": rsi_trend
                }
            
            # Add MACD if available
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd_value = float(latest['MACD_12_26_9'])
                macd_signal = float(latest['MACDs_12_26_9'])
                macd_hist = float(latest['MACDh_12_26_9']) if 'MACDh_12_26_9' in df.columns else macd_value - macd_signal
                macd_trend = "bullish" if macd_value > macd_signal else "bearish"
                indicators["macd"] = {
                    "value": macd_value,
                    "signal": macd_signal,
                    "histogram": macd_hist,
                    "interpretation": macd_trend
                }
            
            # Add Bollinger Bands if available
            if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
                bb_lower = float(latest['BBL_20_2.0'])
                bb_middle = float(latest['BBM_20_2.0'])
                bb_upper = float(latest['BBU_20_2.0'])
                current_price = float(latest['close'])
                
                # Determine position within bands
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
                
                bb_trend = "upper_band" if bb_position > 0.8 else "lower_band" if bb_position < 0.2 else "middle_band"
                
                indicators["bollinger_bands"] = {
                    "lower": bb_lower,
                    "middle": bb_middle,
                    "upper": bb_upper,
                    "width": bb_width,
                    "position": bb_position,
                    "interpretation": bb_trend
                }
            
            # Add Moving Averages if available
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                sma_50 = float(latest['sma_50'])
                sma_200 = float(latest['sma_200'])
                current_price = float(latest['close'])
                
                # Determine trend based on moving averages
                ma_trend = "bullish" if sma_50 > sma_200 and current_price > sma_50 else \
                          "bearish" if sma_50 < sma_200 and current_price < sma_50 else "neutral"
                
                indicators["moving_averages"] = {
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "interpretation": ma_trend
                }
            
            # Prepare signals if available
            signals = {}
            if "signal" in df.columns:
                signal_value = float(latest["signal"])
                signal_text = "strong_buy" if signal_value > 0.6 else \
                             "buy" if signal_value > 0 else \
                             "strong_sell" if signal_value < -0.6 else \
                             "sell" if signal_value < 0 else "neutral"
                signals["overall"] = {
                    "value": signal_value,
                    "interpretation": signal_text
                }
            
            # Prepare predictions if available
            predictions = {}
            if "prediction" in df.columns:
                pred_value = int(latest["prediction"])
                pred_prob = float(latest.get("prediction_probability", 0.5))
                predictions["model"] = {
                    "value": pred_value,
                    "probability": pred_prob,
                    "interpretation": "bullish" if pred_value == 1 else "bearish"
                }
            
            # Combine all data
            market_data = {
                "symbol": df.attrs.get("symbol", "Unknown"),
                "timeframe": df.attrs.get("timeframe", "Unknown"),
                "last_updated": str(df.index[-1]),
                "current_price": float(latest["close"]),
                "recent_prices": recent_prices,
                "indicators": indicators,
                "signals": signals,
                "predictions": predictions
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return {}

    def _generate_prompt(self, market_data):
        """
        Generate a prompt for the LLM based on market data.

        Args:
            market_data (dict): Dictionary with market data

        Returns:
            str: Prompt for the LLM
        """
        try:
            # Format the market data as a readable prompt
            symbol = market_data.get("symbol", "Unknown")
            timeframe = market_data.get("timeframe", "Unknown")
            current_price = market_data.get("current_price", 0)
            
            prompt = f"""You are a professional cryptocurrency trading analyst. Analyze the following market data for {symbol} on {timeframe} timeframe and provide trading recommendations.

CURRENT MARKET DATA:
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: {current_price}
Last Updated: {market_data.get("last_updated", "Unknown")}

"""
            
            # Add indicators section if available
            indicators = market_data.get("indicators", {})
            if indicators:
                prompt += "TECHNICAL INDICATORS:\n"
                
                # Add RSI
                if "rsi" in indicators:
                    rsi = indicators["rsi"]
                    prompt += f"RSI (14): {rsi['value']:.2f} - {rsi['interpretation'].upper()}\n"
                
                # Add MACD
                if "macd" in indicators:
                    macd = indicators["macd"]
                    prompt += f"MACD: {macd['value']:.6f}, Signal: {macd['signal']:.6f}, Histogram: {macd['histogram']:.6f} - {macd['interpretation'].upper()}\n"
                
                # Add Bollinger Bands
                if "bollinger_bands" in indicators:
                    bb = indicators["bollinger_bands"]
                    prompt += f"Bollinger Bands: Lower: {bb['lower']:.2f}, Middle: {bb['middle']:.2f}, Upper: {bb['upper']:.2f}, Width: {bb['width']:.2f} - {bb['interpretation'].upper()}\n"
                
                # Add Moving Averages
                if "moving_averages" in indicators:
                    ma = indicators["moving_averages"]
                    prompt += f"Moving Averages: SMA 50: {ma['sma_50']:.2f}, SMA 200: {ma['sma_200']:.2f} - {ma['interpretation'].upper()}\n"
                
                prompt += "\n"
            
            # Add signals section if available
            signals = market_data.get("signals", {})
            if signals and "overall" in signals:
                signal = signals["overall"]
                prompt += f"OVERALL SIGNAL: {signal['interpretation'].upper()} ({signal['value']:.2f})\n\n"
            
            # Add predictions section if available
            predictions = market_data.get("predictions", {})
            if predictions and "model" in predictions:
                pred = predictions["model"]
                prompt += f"MODEL PREDICTION: {pred['interpretation'].upper()} (Confidence: {pred['probability']:.2f})\n\n"
            
            # Add instructions for the analysis
            prompt += """Based on the above data, provide a comprehensive analysis including:

1. MARKET SUMMARY: A brief summary of the current market conditions.
2. TECHNICAL ANALYSIS: Interpretation of the technical indicators and what they suggest.
3. ENTRY/EXIT LEVELS: Suggested entry points, take profit levels, and stop loss levels with specific price targets.
4. RISK ASSESSMENT: Evaluate the risk level (LOW, MEDIUM, HIGH) and explain why.
5. RECOMMENDATION: A clear trading recommendation (STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL).

Format your response in a clear, structured manner with these sections.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return f"Analyze the market data for {market_data.get('symbol', 'Unknown')} and provide trading recommendations."

    def _call_ollama(self, prompt):
        """
        Call Ollama API with the given prompt.

        Args:
            prompt (str): The prompt to send to Ollama

        Returns:
            str: Ollama's response
        """
        try:
            # Prepare the request
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 1024
                }
            }
            
            # Make the request
            logger.info(f"Calling Ollama API with model {self.model_name}")
            response = requests.post(self.api_url, json=data)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Ollama API returned status code {response.status_code}."
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: Failed to call Ollama API. {str(e)}"

    def _parse_recommendation(self, ollama_response):
        """
        Parse the recommendation from the Ollama response.

        Args:
            ollama_response (str): The response from Ollama

        Returns:
            dict: Structured recommendation
        """
        try:
            # Extract recommendation
            recommendation = "NEUTRAL"  # Default
            if "STRONG BUY" in ollama_response.upper():
                recommendation = "STRONG BUY"
            elif "BUY" in ollama_response.upper():
                recommendation = "BUY"
            elif "STRONG SELL" in ollama_response.upper():
                recommendation = "STRONG SELL"
            elif "SELL" in ollama_response.upper():
                recommendation = "SELL"
            
            # Extract risk assessment
            risk = "MEDIUM"  # Default
            if "RISK: LOW" in ollama_response.upper() or "LOW RISK" in ollama_response.upper():
                risk = "LOW"
            elif "RISK: HIGH" in ollama_response.upper() or "HIGH RISK" in ollama_response.upper():
                risk = "HIGH"
            
            # Extract entry/exit levels if available
            entry_level = None
            take_profit = None
            stop_loss = None
            
            # Look for entry level
            import re
            entry_matches = re.findall(r"ENTRY:?\s*\$?(\d+\.?\d*)", ollama_response, re.IGNORECASE)
            if entry_matches:
                entry_level = float(entry_matches[0])
            
            # Look for take profit level
            tp_matches = re.findall(r"TAKE PROFIT:?\s*\$?(\d+\.?\d*)", ollama_response, re.IGNORECASE)
            if tp_matches:
                take_profit = float(tp_matches[0])
            
            # Look for stop loss level
            sl_matches = re.findall(r"STOP LOSS:?\s*\$?(\d+\.?\d*)", ollama_response, re.IGNORECASE)
            if sl_matches:
                stop_loss = float(sl_matches[0])
            
            # Simple structure for the recommendation
            result = {
                "recommendation": recommendation,
                "risk": risk,
                "entry_level": entry_level,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "analysis": ollama_response,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Ollama recommendation: {e}")
            return {
                "recommendation": "ERROR",
                "risk": "UNKNOWN",
                "analysis": ollama_response,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def analyze(self, df):
        """
        Analyze market data using Ollama.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data and indicators

        Returns:
            dict: Analysis results
        """
        try:
            # Check if Ollama is available
            if not self.check_ollama_available():
                return {"error": "Ollama not available or model not installed"}
            
            # Prepare market data
            market_data = self._prepare_market_data(df)
            if not market_data:
                return {"error": "Failed to prepare market data"}
            
            # Generate prompt
            prompt = self._generate_prompt(market_data)
            
            # Call Ollama
            ollama_response = self._call_ollama(prompt)
            
            # Parse recommendation
            recommendation = self._parse_recommendation(ollama_response)
            
            # Add to DataFrame if needed
            if not df.empty:
                df.attrs["ollama_analysis"] = recommendation
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error performing Ollama analysis: {e}")
            return {"error": str(e)}

    def save_analysis(self, analysis, symbol, timeframe):
        """
        Save analysis results to a file.

        Args:
            analysis (dict): Analysis results
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the analysis

        Returns:
            str: Path to the saved file, or None if saving failed
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join("results", "ollama_analysis")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved Ollama analysis to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving Ollama analysis: {e}")
            return None
