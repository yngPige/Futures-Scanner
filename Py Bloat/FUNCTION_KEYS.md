# Function Key Support in 3lacks Scanner

3lacks Scanner supports using function keys (F1-F4) to toggle settings from anywhere in the application. This document explains how to enable and use this feature.

## Function Keys vs. Letter Keys

The application supports two ways to toggle settings:

1. **Function Keys (F1-F4)**: Requires the keyboard module to be installed
2. **Letter Keys (g, s, t, l)**: Always available as a fallback option

## Installing the Keyboard Module

The keyboard module is required for function key support. It requires administrative privileges to install on Windows.

### Windows

1. Run the `install_keyboard.bat` file by double-clicking it
2. If prompted, allow the script to run with administrative privileges
3. Follow the on-screen instructions

### Other Platforms

1. Open a terminal
2. Navigate to the 3lacks Scanner directory
3. Run the following command:
   ```
   python install_keyboard.py
   ```
4. Follow the on-screen instructions

## Manual Installation

If the installation script doesn't work, you can install the keyboard module manually:

1. Open a command prompt or terminal with administrative privileges
2. Run the following command:
   ```
   pip install keyboard
   ```

## Using Function Keys

Once the keyboard module is installed, you can use the following function keys to toggle settings:

- **F1**: Toggle GPU Acceleration on/off
- **F2**: Toggle Save Results on/off
- **F3**: Toggle Hyperparameter Tuning on/off
- **F4**: Toggle LLM Analysis on/off

## Fallback to Letter Keys

If the keyboard module is not installed, the application will automatically fall back to using letter keys:

- **g**: Toggle GPU Acceleration on/off
- **s**: Toggle Save Results on/off
- **t**: Toggle Hyperparameter Tuning on/off
- **l**: Toggle LLM Analysis on/off

## Troubleshooting

If function keys don't work:

1. Make sure the keyboard module is installed
2. Try running the application with administrative privileges
3. Check if your terminal or console supports function keys
4. Fall back to using letter keys if necessary

## Note for Virtual Environments

If you're using a virtual environment, make sure to install the keyboard module in that environment:

```
# Activate your virtual environment first
pip install keyboard
```
