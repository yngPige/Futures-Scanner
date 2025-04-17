"""
LLM Analysis Module for Crypto Futures Scanner

This module provides functionality to analyze market data using a pretrained LLM model
through Ollama integration.
"""

import requests
import json
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Class to analyze market data using a pretrained LLM model."""

    def __init__(self, model_name="llama3"):
        """
        Initialize the LLM Analyzer.

        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized LLM Analyzer with {model_name} model")

    def _check_ollama_available(self):
        """Check if Ollama is available and the model is loaded."""
        try:
            # First check if Ollama service is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama service is not running. Using fallback analysis.")
                return False
            except requests.exceptions.Timeout:
                logger.warning("Ollama service timed out. Using fallback analysis.")
                return False

            # If we got a response, check if the model is available
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]

                # If no models are loaded, suggest loading the model
                if not models:
                    logger.warning("No models loaded in Ollama. Please load a model using 'ollama pull llama3' or similar command.")
                    return False

                # Check if our specific model is available
                if self.model_name in model_names:
                    logger.info(f"Found model {self.model_name} in Ollama.")
                    return True
                else:
                    # Suggest a model that is available or how to load the requested model
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                    logger.warning(f"You can load the model using 'ollama pull {self.model_name}'")

                    # If there are other models available, we could use one of them instead
                    if models:
                        alternative_model = models[0].get("name")
                        logger.info(f"Using alternative model: {alternative_model}")
                        self.model_name = alternative_model
                        return True
                    return False
            else:
                logger.warning(f"Failed to get models from Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error checking Ollama availability: {e}")
            return False

    def _create_market_analysis_prompt(self, df, symbol, timeframe):
        """
        Create a prompt for market analysis.

        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data

        Returns:
            str: Formatted prompt for the LLM
        """
        # Get the latest data point and some historical context
        latest = df.iloc[-1]

        # Get the last 5 candles for context
        recent_data = df.iloc[-5:].copy()

        # Format the candles data
        candles_text = ""
        for idx, row in recent_data.iterrows():
            candle_time = idx.strftime("%Y-%m-%d %H:%M")
            candle_text = f"Time: {candle_time}, Open: {row['open']:.5f}, High: {row['high']:.5f}, Low: {row['low']:.5f}, Close: {row['close']:.5f}, Volume: {row.get('volume', 'N/A')}"
            candles_text += candle_text + "\n"

        # Collect key indicators
        indicators = {}

        # Price and volume
        indicators["Current Price"] = f"{latest['close']:.5f}"
        if 'volume' in latest:
            indicators["Volume"] = f"{latest['volume']:.2f}"

        # Trend indicators
        if 'sma_20' in df.columns and 'sma_50' in df.columns and 'sma_200' in df.columns:
            indicators["SMA 20"] = f"{latest['sma_20']:.5f}"
            indicators["SMA 50"] = f"{latest['sma_50']:.5f}"
            indicators["SMA 200"] = f"{latest['sma_200']:.5f}"

            # Determine trend based on SMAs
            if latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
                trend = "Strong Uptrend"
            elif latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
                trend = "Strong Downtrend"
            elif latest['sma_20'] > latest['sma_50'] and latest['sma_50'] < latest['sma_200']:
                trend = "Potential Uptrend Reversal"
            elif latest['sma_20'] < latest['sma_50'] and latest['sma_50'] > latest['sma_200']:
                trend = "Potential Downtrend Reversal"
            else:
                trend = "Mixed Signals"

            indicators["Trend"] = trend

        # Momentum indicators
        if 'rsi_14' in df.columns:
            rsi = latest['rsi_14']
            indicators["RSI (14)"] = f"{rsi:.2f}"

            if rsi < 30:
                indicators["RSI Status"] = "Oversold"
            elif rsi > 70:
                indicators["RSI Status"] = "Overbought"
            else:
                indicators["RSI Status"] = "Neutral"

        # MACD
        if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
            macd = latest['MACD_12_26_9']
            signal = latest['MACDs_12_26_9']
            macd_hist = macd - signal

            indicators["MACD"] = f"{macd:.5f}"
            indicators["MACD Signal"] = f"{signal:.5f}"
            indicators["MACD Histogram"] = f"{macd_hist:.5f}"

            if macd > signal:
                indicators["MACD Status"] = "Bullish"
            else:
                indicators["MACD Status"] = "Bearish"

        # Bollinger Bands
        if all(col in df.columns for col in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
            lower = latest['BBL_20_2.0']
            middle = latest['BBM_20_2.0']
            upper = latest['BBU_20_2.0']
            price = latest['close']

            indicators["Bollinger Lower"] = f"{lower:.5f}"
            indicators["Bollinger Middle"] = f"{middle:.5f}"
            indicators["Bollinger Upper"] = f"{upper:.5f}"

            if price < lower:
                indicators["Bollinger Status"] = "Below Lower Band (Potential Buy)"
            elif price > upper:
                indicators["Bollinger Status"] = "Above Upper Band (Potential Sell)"
            else:
                indicators["Bollinger Status"] = "Within Bands"

        # Format indicators as text
        indicators_text = "\n".join([f"{key}: {value}" for key, value in indicators.items()])

        # Create the prompt
        prompt = f"""You are a professional cryptocurrency market analyst. Analyze the following market data for {symbol} on {timeframe} timeframe and provide a concise trading analysis.

MARKET DATA:
{candles_text}

KEY INDICATORS:
{indicators_text}

Based on this data, provide a brief analysis covering:
1. Market sentiment (bullish, bearish, or neutral)
2. Key support and resistance levels
3. Potential entry and exit points
4. Risk assessment (low, medium, high)
5. Short-term price prediction with reasoning

Keep your analysis concise, professional, and data-driven. Focus on actionable insights.
"""
        return prompt

    def analyze_market(self, df, symbol, timeframe):
        """
        Analyze market data using the LLM model.

        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data

        Returns:
            dict: Analysis results from the LLM or fallback analysis if LLM is unavailable
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided, cannot perform LLM analysis")
            return {"error": "No data available for analysis"}

        # Check if Ollama is available
        if not self._check_ollama_available():
            logger.info("Ollama not available or model not loaded, using fallback analysis")
            # Use the fallback analysis instead
            return self.get_fallback_analysis(df, symbol, timeframe)

        try:
            # Create the prompt
            prompt = self._create_market_analysis_prompt(df, symbol, timeframe)

            # Prepare the request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            # Log the request
            logger.info(f"Sending request to Ollama for {symbol} analysis")

            # Send the request
            response = requests.post(self.api_url, json=payload)

            if response.status_code == 200:
                # Parse the response
                result = response.json()
                analysis_text = result.get("response", "")

                # Extract sentiment from the analysis
                sentiment = "neutral"
                if "bullish" in analysis_text.lower():
                    sentiment = "bullish"
                elif "bearish" in analysis_text.lower():
                    sentiment = "bearish"

                # Extract risk assessment
                risk = "medium"
                if "low risk" in analysis_text.lower():
                    risk = "low"
                elif "high risk" in analysis_text.lower():
                    risk = "high"

                # Return the analysis results
                return {
                    "analysis": analysis_text,
                    "sentiment": sentiment,
                    "risk": risk,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "analysis": "LLM analysis failed. Please check the logs for details."
                }

        except Exception as e:
            logger.error(f"Error performing LLM analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "analysis": "LLM analysis encountered an error. Please check the logs for details."
            }

    def get_fallback_analysis(self, df, symbol, timeframe):
        """
        Provide a fallback analysis when the LLM is not available.

        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            symbol (str): Trading symbol
            timeframe (str): Timeframe of the data

        Returns:
            dict: Basic analysis based on technical indicators
        """
        if df is None or df.empty:
            return {"analysis": "No data available for analysis", "is_fallback": True}

        try:
            # Get the latest data point
            latest = df.iloc[-1]

            # Determine basic sentiment based on technical indicators
            sentiment = "neutral"
            reasoning = []

            # Check RSI
            if 'rsi_14' in df.columns:
                rsi = latest['rsi_14']
                if rsi < 30:
                    sentiment = "bullish"
                    reasoning.append(f"RSI is oversold at {rsi:.2f}")
                elif rsi > 70:
                    sentiment = "bearish"
                    reasoning.append(f"RSI is overbought at {rsi:.2f}")
                else:
                    reasoning.append(f"RSI is neutral at {rsi:.2f}")

            # Check MACD
            if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
                macd = latest['MACD_12_26_9']
                signal = latest['MACDs_12_26_9']

                if macd > signal:
                    # If RSI is not bearish, set to bullish
                    if sentiment != "bearish":
                        sentiment = "bullish"
                    reasoning.append("MACD is above signal line (bullish)")
                else:
                    # If RSI is not bullish, set to bearish
                    if sentiment != "bullish":
                        sentiment = "bearish"
                    reasoning.append("MACD is below signal line (bearish)")

            # Check moving averages
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                sma_50 = latest['sma_50']
                sma_200 = latest['sma_200']

                if sma_50 > sma_200:
                    reasoning.append("Price is above 200 SMA (bullish trend)")
                    if sentiment == "neutral":
                        sentiment = "bullish"
                else:
                    reasoning.append("Price is below 200 SMA (bearish trend)")
                    if sentiment == "neutral":
                        sentiment = "bearish"

            # Create a basic analysis text
            analysis_text = f"Technical Analysis for {symbol} ({timeframe}):\n\n"
            analysis_text += f"Market Sentiment: {sentiment.capitalize()}\n\n"
            analysis_text += "Key Observations:\n"
            for i, reason in enumerate(reasoning, 1):
                analysis_text += f"{i}. {reason}\n"

            analysis_text += f"\nCurrent Price: {latest['close']:.5f}\n"

            if 'sma_20' in df.columns:
                analysis_text += f"20 SMA: {latest['sma_20']:.5f}\n"

            if 'sma_50' in df.columns:
                analysis_text += f"50 SMA: {latest['sma_50']:.5f}\n"

            if 'sma_200' in df.columns:
                analysis_text += f"200 SMA: {latest['sma_200']:.5f}\n"

            analysis_text += "\nNote: This is a basic algorithmic analysis. For more detailed insights, ensure Ollama is running with the required model."

            # Always set is_fallback to True for fallback analysis
            return {
                "analysis": analysis_text,
                "sentiment": sentiment,
                "risk": "medium",  # Default risk assessment
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_fallback": True
            }

        except Exception as e:
            logger.error(f"Error creating fallback analysis: {e}")
            return {
                "analysis": "Unable to generate analysis due to an error.",
                "error": str(e),
                "sentiment": "neutral",
                "risk": "medium",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_fallback": True
            }
