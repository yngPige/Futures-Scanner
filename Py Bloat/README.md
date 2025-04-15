# 3lacks Futures Scanner - PyBloat Directory

This directory contains all the executables and dependencies for the 3lacks Futures Scanner application.

A Python application that parses cryptocurrency futures markets with a pretrained AI model for optimized technical analysis to provide price movement predictions.

## Features

- Fetches cryptocurrency futures data from various exchanges using CCXT
- Performs technical analysis using pandas-ta with 150+ indicators
- Trains machine learning models to predict price movements
- Generates trading signals based on technical indicators and AI predictions
- Creates interactive visualizations of price data and indicators
- Backtests trading strategies based on predictions
- Provides LLM-based analysis with entry/exit recommendations and risk assessment

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

### Project Structure

The project is organized into several directories:

- `src/`: Source code
- `utils/`: Utility functions
- `docs/`: Documentation
- `tests/`: Test scripts
- `assets/`: Asset files

### Utility Functions

Utility functions are organized by category in the `utils` directory:

- `utils/data/`: Data utilities (downloading models, etc.)
- `utils/models/`: Model utilities (fixing models, etc.)
- `utils/build/`: Build utilities (building executables, etc.)
- `utils/setup/`: Setup utilities (installing dependencies, etc.)
- `utils/visualization/`: Visualization utilities (creating distributions, icons, etc.)

To run these utilities, use the `run_utils.py` utility:

```bash
# List all available utilities
python run_utils.py --list

# Run a specific utility
python run_utils.py --category data --module download_llm_model.py
```

For more information, see the [utils/README.md](utils/README.md) file.

### Terminal UI

The application can be run with a user-friendly terminal UI directly from this directory:

```
# From the PyBloat directory
python terminal.py

# Or from the root directory using the launcher
python launch_scanner.py
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
python main.py --mode fetch --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000 --save
```

### Analyze Data

```
python main.py --mode analyze --exchange kraken --symbol BTC/USDT --timeframe 1h --save
```

### Train Model

```
python main.py --mode train --exchange kraken --symbol BTC/USDT --timeframe 1h --model-type random_forest --tune --save
```

### Make Predictions

```
python main.py --mode predict --exchange kraken --symbol BTC/USDT --timeframe 1h --model-path models/your_model.joblib --save
```

### Backtest

```
python main.py --mode backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --model-path models/your_model.joblib --save
```

### Run All

```
python main.py --mode all --exchange kraken --symbol BTC/USDT --timeframe 1h --model-type random_forest --save
```

### LLM Analysis

```
python main.py --mode llm --exchange kraken --symbol BTC/USDT --timeframe 1h --use-llm --llm-model mistral-7b --save
```

## Command Line Arguments

- `--mode`: Operation mode (fetch, analyze, train, predict, backtest, all, llm)
- `--exchange`: Exchange to fetch data from (default: kraken)
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
- `--use-llm`: Enable LLM analysis
- `--llm-model`: LLM model to use (default: llama3-8b)
- `--use-gpu`: Use GPU for LLM inference

## PyBloat Directory Structure

This directory contains all the executables and dependencies for the 3lacks Futures Scanner application:

```
Py Bloat/
├── main.py                  # Main application file
├── terminal.py              # Terminal UI application
├── download_llm_model.py    # LLM model downloader
├── fix_model_*.py           # Model fix scripts
├── build_*.py               # Build scripts
├── create_*.py              # Distribution creation scripts
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
    │   ├── technical_analysis.py  # Technical analysis functionality
    │   └── local_llm.py     # LLM analysis functionality
    ├── models/              # AI model module
    │   └── prediction_model.py  # Prediction model functionality
    ├── visualization/       # Visualization module
    │   ├── chart_generator.py  # Chart generation functionality
    │   └── terminal_chart.py  # Terminal-based chart visualization
    ├── ui/                  # User interface module
    │   └── terminal_ui.py   # Terminal UI functionality
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

## LLM Analysis

The application includes a powerful LLM (Large Language Model) analysis feature that provides:

- Trading recommendations based on technical analysis
- Entry and exit price suggestions
- Stop loss and take profit levels
- Risk assessment and risk/reward ratio
- Detailed market analysis and reasoning

### Available LLM Models

The application supports several LLM models that can be downloaded and used locally:

- **Llama 3 8B Instruct**: General purpose model with good reasoning capabilities (4.37 GB)
- **Phi-3 Mini**: Smaller model with good performance for technical analysis (1.91 GB)
- **Mistral 7B**: Good balance of size and performance for financial analysis (3.83 GB)

### Downloading LLM Models

To use the LLM analysis feature, you need to download at least one of the supported models:

```
python download_llm_model.py list
python download_llm_model.py download <model_name>
```

For example:
```
python download_llm_model.py download mistral-7b
```

### Fallback Mode

If no LLM model is available, the application will use a fallback mode that generates recommendations based on technical indicators. This ensures you can still get trading suggestions even without downloading the large LLM models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical analysis indicators
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange API access
- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data access
- [scikit-learn](https://scikit-learn.org/) for machine learning functionality
- [plotly](https://plotly.com/) and [matplotlib](https://matplotlib.org/) for data visualization
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local LLM inference
- [TheBloke](https://huggingface.co/TheBloke) for quantized LLM models
- [plotext](https://github.com/piccolomo/plotext) for terminal-based charts

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
