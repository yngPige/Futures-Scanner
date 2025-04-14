"""
Local LLM Module for Crypto Futures Scanner

This module provides functionality to analyze cryptocurrency data using locally-run
Large Language Models to generate trading recommendations and market insights.
"""

import os
import json
import logging
import tempfile
import platform
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

# Configure logging - only show errors
logging.basicConfig(
    level=logging.ERROR,  # Only log errors and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available models
AVAILABLE_MODELS = {
    "llama3-8b": {
        "name": "llama-3-8b-instruct.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
        "size_gb": 4.37,
        "description": "Llama 3 8B Instruct model (4-bit quantized, medium quality)",
        "trading_focus": "General purpose with good reasoning capabilities"
    },
    "mistral-7b": {
        "name": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 3.83,
        "description": "Mistral 7B Instruct v0.2 model (4-bit quantized, medium quality)",
        "trading_focus": "Good balance of size and performance for financial analysis"
    },
    "phi2": {
        "name": "phi-2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "size_gb": 1.35,
        "description": "Phi-2 model (4-bit quantized, medium quality)",
        "trading_focus": "Smallest model, good for basic analysis on limited hardware"
    },
    "tinyllama": {
        "name": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.6,
        "description": "TinyLlama 1.1B Chat model (4-bit quantized, medium quality)",
        "trading_focus": "Very small model, basic analysis capabilities, runs on any hardware"
    }
}

# Default model settings
DEFAULT_MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
DEFAULT_MODEL_NAME = "phi-2.Q4_K_M.gguf"
DEFAULT_MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"

# Required dependencies
REQUIRED_PACKAGES = {
    "llama-cpp-python": "llama_cpp",
    "huggingface-hub": "huggingface_hub",
    "requests": "requests",
    "tqdm": "tqdm"
}

def check_dependencies():
    """
    Check if all required dependencies are installed.

    Returns:
        tuple: (all_installed, missing_packages)
            - all_installed (bool): True if all dependencies are installed
            - missing_packages (list): List of missing package names
    """
    missing_packages = []

    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    return len(missing_packages) == 0, missing_packages

def install_dependencies(missing_packages=None):
    """
    Install missing dependencies.

    Args:
        missing_packages (list, optional): List of package names to install.
            If None, will check and install all required packages.

    Returns:
        bool: True if installation was successful, False otherwise
    """
    if missing_packages is None:
        _, missing_packages = check_dependencies()

    if not missing_packages:
        logger.info("All required dependencies are already installed.")
        return True

    logger.info(f"Installing missing dependencies: {', '.join(missing_packages)}")

    try:
        # Use subprocess to run pip install
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Successfully installed all dependencies.")
            return True
        else:
            logger.error(f"Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

class LocalLLMAnalyzer:
    """Class to perform LLM-based analysis on cryptocurrency data using local models."""

    def __init__(self, model_name=None, model_path=None, n_ctx=4096, n_gpu_layers=0):
        """
        Initialize the Local LLM Analyzer.

        Args:
            model_name (str, optional): Name of the model to use. If None, will use default.
            model_path (str, optional): Path to the model file or directory. If None, will use default.
            n_ctx (int): Context window size (default: 4096)
            n_gpu_layers (int): Number of layers to offload to GPU (default: 0, CPU only)
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLM model."""
        # Check for required dependencies first
        all_installed, missing_packages = check_dependencies()
        if not all_installed:
            logger.warning(f"Missing required dependencies: {', '.join(missing_packages)}")
            logger.info("LLM functionality requires additional packages.")
            self.llm = None
            return

        try:
            # Import here to avoid dependency issues if not using this module
            from llama_cpp import Llama

            # Check if model file exists, if not download it
            model_file_path = self._get_model_file_path()
            if not os.path.exists(model_file_path):
                download_success = self._download_model(model_file_path)
                if not download_success:
                    logger.error("Failed to download model. Cannot proceed with LLM analysis.")
                    self.llm = None
                    return

            # Load the model
            logger.info(f"Loading model from {model_file_path}")

            # Check if the file exists and has content
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found: {model_file_path}")
                self.llm = None
                return

            if os.path.getsize(model_file_path) == 0:
                logger.error(f"Model file is empty: {model_file_path}")
                self.llm = None
                return

            try:
                self.llm = Llama(
                    model_path=model_file_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.llm = None

        except ImportError as e:
            logger.error(f"Failed to import llama_cpp: {e}")
            logger.info("Please install llama-cpp-python: pip install llama-cpp-python")
            self.llm = None
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.llm = None

    def _get_model_file_path(self):
        """Get the full path to the model file.

        Returns:
            str: Full path to the model file
        """
        # Get the model info from AVAILABLE_MODELS if possible
        for model_key, model_info in AVAILABLE_MODELS.items():
            if model_info.get('name') == self.model_name or model_key == self.model_name:
                # Use the name from the model info
                model_filename = model_info.get('name')
                return os.path.join(self.model_path, model_filename)

        # Fallback to just using the model name directly
        return os.path.join(self.model_path, self.model_name)

    def _download_model(self, model_file_path):
        """
        Download the model file if it doesn't exist.

        Args:
            model_file_path (str): Path where the model should be saved
        """
        try:
            # Find the model URL from AVAILABLE_MODELS
            model_url = None
            model_size = None
            model_key = None

            # Try to find the model in AVAILABLE_MODELS
            for key, info in AVAILABLE_MODELS.items():
                if info.get('name') == self.model_name or key == self.model_name:
                    model_url = info.get('url')
                    model_size = info.get('size_gb')
                    model_key = key
                    break

            if model_url is None:
                # Fallback to default URL if model not found
                model_url = DEFAULT_MODEL_URL
                model_size = 4.37  # Default size for Llama 3 8B
                model_key = "llama3-8b"

            logger.info(f"Model not found at {model_file_path}")
            logger.info(f"Downloading model {model_key} ({model_size} GB)...")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

            # Import required modules
            import requests
            from tqdm import tqdm

            # Download with progress bar
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(model_file_path, 'wb') as f, tqdm(
                desc=self.model_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024*1024):
                    size = f.write(data)
                    bar.update(size)

            logger.info(f"Model downloaded to {model_file_path}")

            # Verify the file was downloaded correctly
            if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
                logger.info(f"Successfully downloaded model: {model_file_path}")
                return True
            else:
                logger.error(f"Downloaded file is empty or does not exist: {model_file_path}")
                return False

        except ImportError as e:
            logger.error(f"Failed to import required module: {e}")
            logger.info("Please run: pip install requests tqdm huggingface-hub")
            return False
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False

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
                    "open": float(latest["open"]),
                    "high": float(latest["high"]),
                    "low": float(latest["low"]),
                    "close": float(latest["close"]),
                    "volume": float(latest["volume"])
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
        # Create a system prompt that instructs the model to act as a financial analyst
        system_prompt = """You are a professional cryptocurrency trading analyst with expertise in technical analysis.
Your task is to analyze market data and provide a detailed trading recommendation."""

        # Create a user prompt with the market data
        user_prompt = f"""
Analyze the following market data for {market_data.get('symbol', 'Unknown')} on the {market_data.get('timeframe', 'Unknown')} timeframe and provide a detailed trading recommendation with specific entry and exit levels.

Current price: {market_data.get('current_price', 'Unknown')}
Last updated: {market_data.get('last_updated', 'Unknown')}

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
5. ENTRY PRICE: Specific price level to enter the trade
6. STOP LOSS: Specific price level to set stop loss
7. TAKE PROFIT: Specific price level(s) for taking profit
8. RISK/REWARD RATIO: Calculate the risk/reward ratio based on your entry, stop loss, and take profit levels

Format your response as a structured analysis with clear sections. Be decisive in your recommendation and provide EXACT price levels for entry, stop loss, and take profit. Label these sections clearly as "ENTRY PRICE:", "STOP LOSS:", and "TAKE PROFIT:" so they can be easily identified.
"""

        # Format for Llama 3 chat format
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"

        return prompt

    def _call_local_llm(self, prompt):
        """
        Call the local LLM with the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM

        Returns:
            str: The LLM's response
        """
        if self.llm is None:
            logger.error("LLM not initialized")
            return "Error: LLM not initialized. Please check the logs for details."

        try:
            # Generate response
            logger.info("Generating response from local LLM...")

            # Set generation parameters
            max_tokens = 1024
            temperature = 0.3
            top_p = 0.9

            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["<|user|>", "<|system|>"],  # Stop tokens for Llama 3 chat format
                echo=False
            )

            # Extract the generated text
            generated_text = response["choices"][0]["text"]

            return generated_text

        except Exception as e:
            logger.error(f"Error calling local LLM: {e}")
            return f"Error: Failed to call local LLM. {str(e)}"

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

            # Extract entry price
            entry_price = None
            entry_match = re.search(r"ENTRY\s*PRICE\s*:?\s*\$?(\d+[.,]\d+)", llm_response, re.IGNORECASE)
            if entry_match:
                try:
                    entry_price = float(entry_match.group(1).replace(',', ''))
                except ValueError:
                    pass

            # Extract stop loss
            stop_loss = None
            stop_match = re.search(r"STOP\s*LOSS\s*:?\s*\$?(\d+[.,]\d+)", llm_response, re.IGNORECASE)
            if stop_match:
                try:
                    stop_loss = float(stop_match.group(1).replace(',', ''))
                except ValueError:
                    pass

            # Extract take profit
            take_profit = None
            profit_match = re.search(r"TAKE\s*PROFIT\s*:?\s*\$?(\d+[.,]\d+)", llm_response, re.IGNORECASE)
            if profit_match:
                try:
                    take_profit = float(profit_match.group(1).replace(',', ''))
                except ValueError:
                    pass

            # Extract risk/reward ratio
            risk_reward = None
            rr_match = re.search(r"RISK[/\s]*REWARD\s*RATIO\s*:?\s*(\d+[.,]?\d*)[:\s]*(\d+[.,]?\d*)", llm_response, re.IGNORECASE)
            if rr_match:
                try:
                    risk_part = float(rr_match.group(1).replace(',', ''))
                    reward_part = float(rr_match.group(2).replace(',', ''))
                    risk_reward = f"{risk_part}:{reward_part}"
                except ValueError:
                    pass
            else:
                # Try alternative format (e.g., "2.5")
                rr_match = re.search(r"RISK[/\s]*REWARD\s*RATIO\s*:?\s*(\d+[.,]\d*)", llm_response, re.IGNORECASE)
                if rr_match:
                    try:
                        risk_reward = float(rr_match.group(1).replace(',', ''))
                    except ValueError:
                        pass

            # Simple structure for the recommendation
            result = {
                "recommendation": recommendation,
                "risk": risk,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": risk_reward,
                "analysis": llm_response,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }

            return result

        except Exception as e:
            logger.error(f"Error parsing LLM recommendation: {e}")
            return {
                "recommendation": "ERROR",
                "risk": "UNKNOWN",
                "analysis": llm_response,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }

    def _generate_fallback_recommendation(self, df):
        """
        Generate a fallback recommendation based on technical indicators when LLM is not available.

        Args:
            df (pd.DataFrame): DataFrame with market data and technical indicators

        Returns:
            dict: Analysis results including trading recommendation
        """
        try:
            # Get the latest data point
            latest_data = df.iloc[-1]

            # Extract key indicators
            rsi = latest_data.get('RSI_14', 50)
            macd = latest_data.get('MACD_12_26_9', 0)
            macd_signal = latest_data.get('MACDs_12_26_9', 0)
            macd_hist = latest_data.get('MACDh_12_26_9', 0)
            ema_short = latest_data.get('EMA_9', 0)
            ema_long = latest_data.get('EMA_21', 0)
            close_price = latest_data.get('close', 0)

            # Determine trend based on EMAs
            trend = "bullish" if ema_short > ema_long else "bearish"

            # Determine signal based on RSI and MACD
            signal = "neutral"
            if rsi < 30 and macd > macd_signal:
                signal = "buy"
            elif rsi > 70 and macd < macd_signal:
                signal = "sell"
            elif macd > macd_signal and macd_hist > 0:
                signal = "buy"
            elif macd < macd_signal and macd_hist < 0:
                signal = "sell"

            # Calculate entry, stop loss, and take profit levels
            entry_price = close_price
            stop_loss = entry_price * 0.95 if signal == "buy" else entry_price * 1.05
            take_profit = entry_price * 1.1 if signal == "buy" else entry_price * 0.9

            # Format the recommendation
            recommendation = {
                "market_summary": f"The market is currently in a {trend} trend.",
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": abs((take_profit - entry_price) / (entry_price - stop_loss)) if signal != "neutral" else 0,
                "confidence": "medium",
                "reasoning": f"Based on technical indicators: RSI at {rsi:.2f}, MACD at {macd:.2f}, with {trend} EMA crossover.",
                "indicators_used": ["RSI", "MACD", "EMA"],
                "generated_by": "fallback_system"
            }

            return recommendation

        except Exception as e:
            logger.error(f"Error generating fallback recommendation: {e}")
            return {
                "market_summary": "Unable to analyze market data.",
                "signal": "neutral",
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "risk_reward_ratio": 0,
                "confidence": "low",
                "reasoning": "Error in fallback analysis system.",
                "indicators_used": [],
                "generated_by": "fallback_system"
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

        # Check if LLM is initialized
        if self.llm is None:
            logger.warning("LLM is not initialized, cannot perform analysis")
            logger.info("Using fallback recommendation based on technical indicators")

            # Generate a fallback recommendation based on technical indicators
            fallback_recommendation = self._generate_fallback_recommendation(df)

            # Add to DataFrame if needed
            if not df.empty:
                df.attrs["llm_analysis"] = fallback_recommendation

            return fallback_recommendation

        try:
            # Prepare market data
            market_data = self._prepare_market_data(df)
            if not market_data:
                return {"error": "Failed to prepare market data"}

            # Generate prompt
            prompt = self._generate_prompt(market_data)

            # Call LLM
            llm_response = self._call_local_llm(prompt)

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


# Available models
AVAILABLE_MODELS = {
    "llama3-8b": {
        "name": "llama-3-8b-instruct.Q4_K_M.gguf",
        "description": "Llama 3 8B Instruct (Quantized 4-bit)",
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
        "size_gb": 4.37,
        "details": "General-purpose LLM with good performance across various tasks. Balanced between size and capability.",
        "trading_focus": "Medium",
        "hardware_req": "8GB+ VRAM GPU recommended, can run on CPU",
        "strengths": "Versatile, good instruction following, reasonable size",
        "weaknesses": "Not specialized for financial analysis"
    },
    "mistral-7b": {
        "name": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B Instruct v0.2 (Quantized 4-bit)",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_gb": 3.83,
        "details": "Efficient model with strong performance for its size. Good balance of capabilities and resource requirements.",
        "trading_focus": "Medium",
        "hardware_req": "6GB+ VRAM GPU recommended, can run on CPU",
        "strengths": "Efficient, good reasoning, smaller size",
        "weaknesses": "Less powerful than larger models for complex analysis"
    },
    "phi2": {
        "name": "phi-2.Q4_K_M.gguf",
        "description": "Phi-2 (Quantized 4-bit)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "size_gb": 1.35,
        "details": "Microsoft's small but capable model. Good performance for its tiny size.",
        "trading_focus": "Low",
        "hardware_req": "4GB RAM is sufficient, runs well on CPU",
        "strengths": "Very small size, fast inference, runs on modest hardware",
        "weaknesses": "Limited context window, less powerful than larger models"
    },
    "tinyllama": {
        "name": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B Chat (Quantized 4-bit)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.6,
        "details": "Ultra-small model that can run on almost any hardware. Limited capabilities but very fast.",
        "trading_focus": "Low",
        "hardware_req": "2GB RAM is sufficient, runs on any modern CPU",
        "strengths": "Extremely small size, very fast inference, minimal resource requirements",
        "weaknesses": "Limited capabilities, basic analysis only"
    }
}


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    import pandas as pd
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

    # Initialize Local LLM analyzer
    llm_analyzer = LocalLLMAnalyzer()

    # Analyze with LLM
    recommendation = llm_analyzer.analyze(data_with_indicators)

    # Print results
    print(json.dumps(recommendation, indent=2))
