# 3lacks Futures Scanner

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

### Running the Application

The application executables are located in the `Py Bloat` directory. To run the application:

1. Use the provided launcher script:
```
python launch_scanner.py
```

2. Or use the batch file (Windows):
```
run_scanner.bat
```

This will launch the application from the PyBloat directory, which contains all the necessary executables and dependencies.

### Terminal UI

The application uses a user-friendly terminal UI:

```
# Main menu options
1. Fetch Data
2. Analyze Data
3. Train Model
4. Make Predictions
5. Backtest Strategy
6. Run All Steps
7. LLM Analysis
8. Settings
```

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

## LLM Analysis

The application includes LLM-based analysis capabilities:

1. Download an LLM model (Settings > Change LLM Model)
2. Enable LLM Analysis (Settings > Toggle LLM Analysis)
3. Run LLM Analysis from the main menu

The LLM analysis provides:
- Market sentiment analysis
- Entry/exit recommendations
- Stop loss and take profit levels
- Risk assessment

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the open-source libraries that made this project possible
- Special thanks to the cryptocurrency community for their support
