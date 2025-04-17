# Terminal UI Guide for Crypto Futures Scanner

This guide explains how to use the terminal-based user interface for the Crypto Futures Scanner application.

## Getting Started

To launch the terminal UI, run:

```
python terminal.py
```

## Navigation

The terminal UI is organized into menus that you can navigate using the keyboard:

- Enter the number or letter corresponding to your choice
- Press Enter to confirm your selection
- Use 'b' to go back to the previous menu
- Use 'q' to quit the application from the main menu

## Main Menu

The main menu provides access to all the main functions of the application:

1. **Run Analysis** - Execute all analysis steps in sequence
2. **Fetch Data** - Download cryptocurrency data from exchanges
3. **Train Model** - Train a machine learning model for predictions
4. **Make Predictions** - Use a trained model to make predictions
5. **Backtest Strategy** - Test the performance of the predictions
s. **Change Symbol** - Select the cryptocurrency pair to analyze
6. **Settings** - Configure application settings
c. **Clear Data** - Clear cached data
h. **How to Use** - Display help information
q. **Quit** - Exit the application

## Settings Menu

The settings menu allows you to configure various parameters:

1. **Change Timeframe** - Set the time interval for the data
2. **Change Data Limit** - Set the number of candles to fetch
3. **Change Exchange** - Select the exchange to fetch data from
4. **Change Model Type** - Choose the machine learning model type
5. **Select Model Path** - Choose a previously trained model
g. **Toggle GPU Acceleration** - Enable/disable GPU acceleration
s. **Toggle Save Results** - Enable/disable saving results to disk
t. **Toggle Hyperparameter Tuning** - Enable/disable model tuning
b. **Back to Main Menu** - Return to the main menu

## Workflow Example

Here's a typical workflow using the terminal UI:

1. Use the "Change Symbol" option (s) in the main menu to select your cryptocurrency
2. Configure other settings if needed (option 6)
3. Run the complete analysis (option 1) or individual steps (options 2-5)

## Tips

- The current settings are always displayed alongside the menu options
- After each operation, you'll see the results and be prompted to press Enter to continue
- If you want to save disk space, disable the "Save Results" option in settings
- For better model performance, enable the "Hyperparameter Tuning" option in settings (but note that this will take longer to train)
- Use GPU acceleration for faster model training if your system supports it

## Troubleshooting

If you encounter any issues:

- Check the log file (error_logs/crypto_scanner.log) for detailed error messages
- Make sure you have all the required dependencies installed
- Verify that your internet connection is working if fetching data from exchanges
- Try using a different symbol or timeframe if the current one is causing issues
