"""
LLM Analysis Module for Crypto Futures Scanner

This module provides functionality to analyze cryptocurrency data using Large Language Models
to generate trading recommendations and market insights.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Class to perform LLM-based analysis on cryptocurrency data."""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the LLM Analyzer.
        
        Args:
            api_key (str, optional): API key for the LLM service. If None, will try to get from environment.
            model (str): The LLM model to use (default: "gpt-3.5-turbo")
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No API key provided. LLM analysis will not be available.")
    
    def _prepare_market_data(self, df):
        """
        Prepare market data for LLM analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            
        Returns:
            dict: Dictionary with formatted market data
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot prepare market data")
            return {}
        
        try:
            # Get the latest data point and a few previous ones for context
            latest = df.iloc[-1]
            recent = df.iloc[-5:].reset_index()
            
            # Format recent price data
            recent_prices = []
            for _, row in recent.iterrows():
                recent_prices.append({
                    "date": str(row["timestamp"] if "timestamp" in row else row.name),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                })
            
            # Prepare key indicators
            indicators = {}
            
            # Trend indicators
            if "ema_12" in df.columns and "ema_26" in df.columns:
                indicators["ema_crossover"] = {
                    "ema_12": float(latest["ema_12"]),
                    "ema_26": float(latest["ema_26"]),
                    "status": "bullish" if latest["ema_12"] > latest["ema_26"] else "bearish"
                }
            
            if "sma_50" in df.columns and "sma_200" in df.columns:
                indicators["sma_crossover"] = {
                    "sma_50": float(latest["sma_50"]),
                    "sma_200": float(latest["sma_200"]),
                    "status": "bullish" if latest["sma_50"] > latest["sma_200"] else "bearish"
                }
            
            # Momentum indicators
            if "rsi_14" in df.columns:
                rsi = float(latest["rsi_14"])
                indicators["rsi"] = {
                    "value": rsi,
                    "status": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                }
            
            # MACD
            if all(col in df.columns for col in ["MACD_12_26_9", "MACDs_12_26_9"]):
                macd = float(latest["MACD_12_26_9"])
                macd_signal = float(latest["MACDs_12_26_9"])
                indicators["macd"] = {
                    "macd": macd,
                    "signal": macd_signal,
                    "histogram": macd - macd_signal,
                    "status": "bullish" if macd > macd_signal else "bearish"
                }
            
            # Bollinger Bands
            if all(col in df.columns for col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]):
                close = float(latest["close"])
                lower = float(latest["BBL_20_2.0"])
                middle = float(latest["BBM_20_2.0"])
                upper = float(latest["BBU_20_2.0"])
                
                bb_status = "middle"
                if close < lower:
                    bb_status = "below_lower"
                elif close > upper:
                    bb_status = "above_upper"
                
                indicators["bollinger_bands"] = {
                    "lower": lower,
                    "middle": middle,
                    "upper": upper,
                    "width": (upper - lower) / middle,
                    "status": bb_status
                }
            
            # Stochastic
            if "stoch_14_3_3" in df.columns and "stochd_14_3_3" in df.columns:
                stoch_k = float(latest["stoch_14_3_3"])
                stoch_d = float(latest["stochd_14_3_3"])
                
                stoch_status = "neutral"
                if stoch_k < 20 and stoch_d < 20:
                    stoch_status = "oversold"
                elif stoch_k > 80 and stoch_d > 80:
                    stoch_status = "overbought"
                
                indicators["stochastic"] = {
                    "k": stoch_k,
                    "d": stoch_d,
                    "status": stoch_status
                }
            
            # Volume indicators
            if "obv" in df.columns:
                indicators["obv"] = {
                    "value": float(latest["obv"])
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
            market_data (dict): Dictionary with formatted market data
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = f"""
You are a professional cryptocurrency trading analyst. Analyze the following market data for {market_data.get('symbol', 'Unknown')} on the {market_data.get('timeframe', 'Unknown')} timeframe and provide a detailed trading recommendation.

Current price: {market_data.get('current_price', 'Unknown')}
Last updated: {market_data.get('last_updated', 'Unknown')}

Recent price action:
{json.dumps(market_data.get('recent_prices', []), indent=2)}

Technical indicators:
{json.dumps(market_data.get('indicators', {}), indent=2)}

Existing signals:
{json.dumps(market_data.get('signals', {}), indent=2)}

Model predictions:
{json.dumps(market_data.get('predictions', {}), indent=2)}

Based on this data, provide:
1. A concise market summary
2. Key technical levels (support, resistance)
3. A clear trading recommendation (Strong Buy, Buy, Neutral, Sell, Strong Sell)
4. Risk assessment (Low, Medium, High)
5. Potential price targets for both bullish and bearish scenarios
6. Suggested stop loss level

Format your response as a structured analysis with clear sections. Be decisive in your recommendation.
"""
        return prompt
    
    def _call_llm_api(self, prompt):
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The LLM's response
        """
        if not self.api_key:
            logger.error("No API key available for LLM analysis")
            return "Error: No API key available for LLM analysis. Please set your OPENAI_API_KEY environment variable."
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a professional cryptocurrency trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}. {response.text}"
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Error: Failed to call LLM API. {str(e)}"
    
    def _parse_recommendation(self, llm_response):
        """
        Parse the LLM response to extract structured recommendation.
        
        Args:
            llm_response (str): The LLM's response
            
        Returns:
            dict: Structured recommendation
        """
        try:
            # Extract trading recommendation
            recommendation = "NEUTRAL"  # Default
            if "STRONG BUY" in llm_response.upper() or "STRONG BULLISH" in llm_response.upper():
                recommendation = "STRONG BUY"
            elif "BUY" in llm_response.upper() or "BULLISH" in llm_response.upper():
                recommendation = "BUY"
            elif "STRONG SELL" in llm_response.upper() or "STRONG BEARISH" in llm_response.upper():
                recommendation = "STRONG SELL"
            elif "SELL" in llm_response.upper() or "BEARISH" in llm_response.upper():
                recommendation = "SELL"
            
            # Extract risk assessment
            risk = "MEDIUM"  # Default
            if "RISK: LOW" in llm_response.upper() or "LOW RISK" in llm_response.upper():
                risk = "LOW"
            elif "RISK: HIGH" in llm_response.upper() or "HIGH RISK" in llm_response.upper():
                risk = "HIGH"
            
            # Simple structure for the recommendation
            result = {
                "recommendation": recommendation,
                "risk": risk,
                "analysis": llm_response,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM recommendation: {e}")
            return {
                "recommendation": "ERROR",
                "risk": "UNKNOWN",
                "analysis": llm_response,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze(self, df):
        """
        Analyze market data using LLM and generate trading recommendation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            
        Returns:
            dict: Trading recommendation and analysis
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot perform LLM analysis")
            return {"error": "Empty DataFrame provided"}
        
        try:
            # Prepare market data
            market_data = self._prepare_market_data(df)
            if not market_data:
                return {"error": "Failed to prepare market data"}
            
            # Generate prompt
            prompt = self._generate_prompt(market_data)
            
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Parse recommendation
            recommendation = self._parse_recommendation(llm_response)
            
            # Add to DataFrame if needed
            if not df.empty:
                df.attrs["llm_analysis"] = recommendation
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error performing LLM analysis: {e}")
            return {"error": str(e)}
    
    def save_analysis(self, recommendation, symbol, timeframe):
        """
        Save LLM analysis to file.
        
        Args:
            recommendation (dict): Trading recommendation and analysis
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the analysis
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/{symbol.replace('/', '_')}_{timeframe}_llm_analysis_{timestamp}.json"
            
            # Save to file
            with open(filename, "w") as f:
                json.dump(recommendation, f, indent=2)
            
            logger.info(f"Saved LLM analysis to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving LLM analysis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    import yfinance as yf
    
    # Download sample data
    data = yf.download('BTC-USD', period='1mo', interval='1d')
    
    # Initialize analyzer
    from src.analysis.technical_analysis import TechnicalAnalyzer
    
    # Add technical indicators
    ta = TechnicalAnalyzer()
    data_with_indicators = ta.add_all_indicators(data)
    
    # Set DataFrame attributes
    data_with_indicators.attrs["symbol"] = "BTC-USD"
    data_with_indicators.attrs["timeframe"] = "1d"
    
    # Initialize LLM analyzer
    llm_analyzer = LLMAnalyzer()
    
    # Analyze with LLM
    recommendation = llm_analyzer.analyze(data_with_indicators)
    
    # Print results
    print(json.dumps(recommendation, indent=2))
