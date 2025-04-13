# Crypto Futures Scanner

A Python application that parses cryptocurrency futures markets with a pretrained AI model for optimized technical analysis to provide price movement predictions.

## Features

- Fetches cryptocurrency futures data from various exchanges using CCXT
- Performs technical analysis using pandas-ta with 150+ indicators
- Trains machine learning models to predict price movements
- Generates trading signals based on technical indicators and AI predictions
- Creates interactive visualizations of price data and indicators
- Backtests trading strategies based on predictions

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/Futures-Scanner.git
cd Futures-Scanner
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

### Terminal UI

The application can be run with a user-friendly terminal UI:

```
python terminal.py
```

This will launch an interactive terminal interface where you can:
- Navigate through menus to access different features
- Configure settings like symbol, timeframe, and model type
- Run operations like fetching data, analyzing, training models, and backtesting
- View results directly in the terminal

### Command Line

Alternatively, the application can be run in different modes via command line:

### Fetch Data

```
python main.py --mode fetch --exchange coinbase --symbol BTC/USDT --timeframe 1h --limit 1000 --save
```

### Analyze Data

```
python main.py --mode analyze --exchange coinbase --symbol BTC/USDT --timeframe 1h --save
```

### Train Model

```
python main.py --mode train --exchange coinbase --symbol BTC/USDT --timeframe 1h --model-type random_forest --tune --save
```

### Make Predictions

```
python main.py --mode predict --exchange coinbase --symbol BTC/USDT --timeframe 1h --model-path models/your_model.joblib --save
```

### Backtest

```
python main.py --mode backtest --exchange coinbase --symbol BTC/USDT --timeframe 1h --model-path models/your_model.joblib --save
```

### Run All Steps

```
python main.py --mode all --exchange coinbase --symbol BTC/USDT --timeframe 1h --model-type random_forest --save
```

## Command Line Arguments

- `--mode`: Operation mode (fetch, analyze, train, predict, backtest, all)
- `--exchange`: Exchange to fetch data from (default: coinbase)
- `--symbol`: Symbol to analyze (default: BTC/USDT)
- `--timeframe`: Timeframe for data (default: 1h)
- `--limit`: Number of candles to fetch (default: 1000)
- `--model-type`: Type of model to use (random_forest, gradient_boosting)
- `--model-path`: Path to saved model
- `--tune`: Tune model hyperparameters
- `--theme`: Chart theme (dark, light)
- `--interactive`: Generate interactive charts
- `--save`: Save results
- `--no-display`: Do not display charts

## Project Structure

```
Futures-Scanner/
├── main.py                  # Main application file
├── requirements.txt         # Required packages
├── README.md                # This file
├── data/                    # Directory for storing data
├── models/                  # Directory for storing trained models
├── results/                 # Directory for storing analysis results
├── charts/                  # Directory for storing generated charts
└── src/                     # Source code
    ├── data/                # Data fetching module
    │   └── data_fetcher.py  # Data fetching functionality
    ├── analysis/            # Technical analysis module
    │   └── technical_analysis.py  # Technical analysis functionality
    ├── models/              # AI model module
    │   └── prediction_model.py  # Prediction model functionality
    ├── visualization/       # Visualization module
    │   └── chart_generator.py  # Chart generation functionality
    └── utils/               # Utility functions
        └── helpers.py       # Helper utilities
```

## Technical Indicators

The application uses pandas-ta to calculate various technical indicators, including:

- Moving Averages (SMA, EMA, etc.)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Ichimoku Cloud
- And many more...

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical analysis indicators
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange API access
- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data access
- [scikit-learn](https://scikit-learn.org/) for machine learning functionality
- [plotly](https://plotly.com/) and [matplotlib](https://matplotlib.org/) for data visualization

## Development Notes

### Git Ignore Rules

The project includes a comprehensive `.gitignore` file to prevent unnecessary files from being tracked in version control:

- **Data files**: All files in the `data/` directory are ignored to avoid storing large datasets in the repository
- **Model files**: Trained models in the `models/` directory are ignored due to their large size
- **Results**: Analysis results, predictions, and backtest results in the `results/` directory are ignored
- **Reports**: HTML reports and other output files are ignored
- **Logs**: Log files are ignored to keep the repository clean
- **Temporary files**: Script files generated by the terminal output generator are ignored

When contributing to this project, make sure to respect these ignore rules and avoid committing any of the ignored file types.

### Cleanup Script

The project includes a cleanup script (`cleanup.py`) to help manage disk space and prepare the project for version control:

```
# Clean up all generated files (use with caution)
python cleanup.py --all

# Show what would be deleted without actually deleting
python cleanup.py --all --dry-run

# Clean up specific types of files
python cleanup.py --data --results --logs
```

Available options:
- `--all`: Clean up all files (data, models, results, reports, logs, temp)
- `--data`: Clean up data files
- `--models`: Clean up model files
- `--results`: Clean up result files
- `--reports`: Clean up report files
- `--logs`: Clean up log files
- `--temp`: Clean up temporary files
- `--dry-run`: Show what would be deleted without actually deleting
