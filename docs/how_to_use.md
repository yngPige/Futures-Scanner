# 3lacks Scanner - User Guide

## Introduction

3lacks Scanner is a powerful cryptocurrency futures analysis tool that combines technical analysis, machine learning, and AI-powered insights to help traders make informed decisions. This guide will walk you through all the features and functions of the application.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yngPige/Futures-Scanner.git
   cd Futures-Scanner
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the application:
   ```
   python terminal.py
   ```

## Main Menu Overview

When you start 3lacks Scanner, you'll see the main menu with the following options:

- **1: Fetch Data** - Download historical price data for a cryptocurrency
- **2: Analyze Data** - Perform technical analysis on the fetched data
- **3: Train Model** - Train a machine learning model for price prediction
- **4: Make Predictions** - Use a trained model to predict future price movements
- **5: Backtest Strategy** - Test a trading strategy on historical data
- **6: Run All** - Execute all the above steps in sequence
- **7: LLM Analysis** - Use AI to analyze market data and provide trading recommendations
- **8: Settings** - Configure application settings
- **q: Quit** - Exit the application

On the right side of the menu, you'll see your current settings, which affect how each function operates.

## Step-by-Step Guides

### 1. Fetching Data

This step downloads historical price data for your selected cryptocurrency.

1. From the main menu, press `1` to select "Fetch Data"
2. The application will connect to the exchange API and download data based on your current settings
3. You'll see a loading animation with logs showing the progress
4. Once complete, a summary of the fetched data will be displayed

**Tips:**
- Make sure your symbol and exchange settings are correct before fetching
- The "limit" setting controls how many candles (time periods) to fetch

### 2. Analyzing Data

This step calculates technical indicators and generates trading signals.

1. From the main menu, press `2` to select "Analyze Data"
2. The application will first fetch data if needed
3. It will then calculate various technical indicators (RSI, MACD, Bollinger Bands, etc.)
4. A detailed analysis report will be displayed showing:
   - Market overview (price, volume)
   - Key indicators with interpretations
   - Overall signal (Buy, Sell, or Neutral)
   - Recent trading signals

**Tips:**
- The analysis includes over 60 different technical indicators
- Results are saved to the `results` folder for later reference

### 3. Training a Model

This step trains a machine learning model to predict future price movements.

1. From the main menu, press `3` to select "Train Model"
2. The application will fetch and analyze data if needed
3. It will then train a machine learning model using the analyzed data
4. Once complete, model information will be displayed
5. The trained model is automatically saved and selected for future predictions

**Tips:**
- You can choose between different model types in Settings
- The "tune" setting enables hyperparameter optimization for better performance
- Trained models are saved to the `models` folder

### 4. Making Predictions

This step uses a trained model to predict future price movements.

1. From the main menu, press `4` to select "Make Predictions"
2. The application will fetch and analyze data if needed
3. It will then use your selected model to make predictions
4. A prediction summary will be displayed showing:
   - Bullish vs. bearish signal counts and percentages
   - Latest prediction with confidence level
   - Current price

**Tips:**
- You must have a trained model selected before making predictions
- Higher confidence levels indicate stronger signals
- Predictions are saved to the `results` folder

### 5. Backtesting a Strategy

This step tests a trading strategy on historical data to evaluate its performance.

1. From the main menu, press `5` to select "Backtest Strategy"
2. The application will fetch data, analyze it, and make predictions if needed
3. It will then simulate trading based on the predicted signals
4. A detailed backtest report will be displayed showing:
   - Performance metrics (profit/loss, max drawdown, etc.)
   - Trading metrics (win rate, number of trades, etc.)
   - Comparison to buy-and-hold strategy

**Tips:**
- Green metrics indicate positive performance, red indicates negative
- The Sharpe ratio measures risk-adjusted returns (higher is better)
- Backtest results are saved to the `results` folder

### 6. Running All

This option executes all the above steps in sequence.

1. From the main menu, press `6` to select "Run All"
2. The application will:
   - Fetch data
   - Perform technical analysis
   - Train a model (if no model is selected)
   - Make predictions
   - Backtest a strategy
   - Run LLM analysis (if enabled)
3. A comprehensive report will be displayed at the end

**Tips:**
- This is the quickest way to get a complete analysis
- Each step's results are saved to their respective folders

### 7. LLM Analysis

This step uses a Large Language Model (LLM) to analyze market data and provide trading recommendations.

1. From the main menu, press `7` to select "LLM Analysis"
2. The application will fetch and analyze data if needed
3. It will then initialize the selected LLM model
4. The model will analyze the market data and generate insights
5. A detailed AI analysis will be displayed showing:
   - Trading recommendation (Buy, Sell, or Hold)
   - Risk assessment
   - Explanation of the reasoning

**Tips:**
- You must enable LLM analysis in Settings first
- The first run may take longer as it downloads the model
- GPU acceleration can significantly speed up analysis if available

## Settings Configuration

To configure the application settings:

1. From the main menu, press `8` to select "Settings"
2. You'll see a list of configurable options:
   - **Symbol**: The cryptocurrency pair to analyze (e.g., BTC/USDT) - Opens a user-friendly popup selector
   - **Timeframe**: The candle interval (e.g., 1h, 4h, 1d)
   - **Limit**: Number of historical candles to fetch
   - **Exchange**: The exchange to fetch data from - Can also be changed in the symbol selector
   - **Model Type**: The machine learning algorithm to use (Random Forest: ensemble of decision trees, robust to overfitting; Gradient Boosting: sequential ensemble with higher accuracy but more prone to overfitting)
   - **Model Path**: Path to a saved model file
   - **Theme**: Visual theme (dark or light)
   - **Save**: Whether to save results to files
   - **Tune**: Whether to optimize model hyperparameters
   - **Use LLM**: Whether to enable AI analysis
   - **LLM Model**: The AI model to use for analysis
   - **Use GPU**: Whether to use GPU acceleration for LLM

3. Select the option you want to change and follow the prompts

### Symbol Selector

The Symbol Selector is a user-friendly popup window that allows you to:

- Browse and search for cryptocurrency symbols
- See which symbols are available on the selected exchange
- Switch between different exchanges
- Get suggestions for alternative symbols when a symbol is not available on an exchange

To use the Symbol Selector:

1. Go to Settings and select "Change Symbol" or "Change Exchange" and press 's'
2. Use arrow keys to navigate the list of symbols
3. Type to search for specific symbols
4. Press Tab to cycle through available exchanges
5. Press Enter to select a symbol
6. If a symbol is not available on the current exchange, you'll be offered alternatives

### Toggle Options

You can toggle certain settings in the Settings menu using the following letter keys:

- **g**: Toggle GPU Acceleration on/off
- **s**: Toggle Save Results on/off
- **t**: Toggle Hyperparameter Tuning on/off
- **l**: Toggle LLM Analysis on/off

When a toggle is enabled, it will be displayed as [ON] in green text in the settings menu.

**Tips:**
- Settings are applied immediately and affect all operations
- The "b" option returns you to the main menu

## Understanding Results

### Model Types

- **Random Forest**: An ensemble of decision trees that makes predictions by averaging the output of multiple trees. It's robust to overfitting and handles non-linear patterns well, making it a good default choice for most trading scenarios.

- **Gradient Boosting**: A sequential ensemble method that builds trees one at a time, with each tree correcting the errors of the previous ones. It often achieves higher accuracy than Random Forest but is more prone to overfitting, especially with noisy data like cryptocurrency prices.

### Technical Analysis Results

- **RSI (Relative Strength Index)**:
  - Below 30: Oversold (potential buy)
  - Above 70: Overbought (potential sell)
  - Between 30-70: Neutral

- **MACD (Moving Average Convergence Divergence)**:
  - MACD above Signal: Bullish
  - MACD below Signal: Bearish

- **Bollinger Bands**:
  - Price below lower band: Potentially oversold
  - Price above upper band: Potentially overbought
  - Price within bands: Neutral

- **Moving Averages**:
  - SMA50 above SMA200: Golden Cross (bullish)
  - SMA50 below SMA200: Death Cross (bearish)

### Prediction Results

- **Bullish signals**: Percentage of candles predicted to go up
- **Bearish signals**: Percentage of candles predicted to go down
- **Confidence**: Higher values indicate stronger conviction in the prediction

### Backtest Results

- **Strategy Return**: Total return of the trading strategy
- **Buy-Hold Return**: Return from simply buying and holding
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest percentage drop from peak to trough
- **Sharpe Ratio**: Risk-adjusted return (higher is better)

### LLM Analysis Results

- **Trading Recommendation**: The AI's suggested action (Buy, Sell, Hold)
- **Risk Assessment**: Evaluation of current market risk
- **Explanation**: The reasoning behind the recommendation

## Troubleshooting

### Common Issues

1. **"Failed to fetch data"**
   - Check your internet connection
   - Verify the symbol exists on the selected exchange
   - Try a different exchange

2. **"Analysis resulted in empty DataFrame"**
   - Try increasing the "limit" setting to fetch more data
   - Check if the symbol has enough historical data

3. **"Failed to train model"**
   - Ensure you have enough data (at least 100 candles)
   - Try a different model type

4. **"LLM analysis failed"**
   - Check if you have enough disk space for the model
   - Try disabling GPU if you're having GPU-related errors
   - Ensure you have a stable internet connection for model download

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the log file at `crypto_scanner.log` for detailed error messages
2. Visit the GitHub repository for updates and known issues
3. Submit an issue on GitHub with details about your problem

## Advanced Usage

### Custom Indicators

The application includes a wide range of technical indicators, but you can add custom ones by modifying the `custom_indicators.py` file.

### Model Tuning

Enable the "tune" setting to automatically optimize model hyperparameters. This takes longer but can significantly improve prediction accuracy.

### GPU Acceleration

If you have a compatible GPU, enable "Use GPU" in settings to speed up LLM analysis. This requires appropriate CUDA drivers to be installed.

---

Thank you for using 3lacks Scanner! We hope this tool helps improve your trading decisions.

Donations accepted but not required:
CGUDnm2vjTthuuxdYv7wJG6r9akxq8ascgsCXB7Dvgjz
