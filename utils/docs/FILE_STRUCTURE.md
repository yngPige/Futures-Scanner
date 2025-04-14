# 3lacks Scanner File Structure

This document explains the file structure of the 3lacks Scanner project.

## Directory Structure

```
3lacks-Scanner/
├── assets/                  # Asset files (icons, images)
├── charts/                  # Generated chart files
├── data/                    # Data files
├── docs/                    # Documentation
├── models/                  # Trained models
├── reports/                 # Generated reports
├── results/                 # Analysis results
├── scripts/                 # Utility scripts
│   ├── build/               # Build scripts
│   │   └── hooks/           # PyInstaller hooks
│   ├── create/              # Distribution creation scripts
│   ├── download/            # Model download scripts
│   └── fix_model/           # Model fix scripts
├── setup/                   # Installation and setup scripts
├── src/                     # Source code
│   ├── analysis/            # Analysis modules
│   ├── data/                # Data handling modules
│   ├── models/              # Model definitions
│   ├── ui/                  # User interface modules
│   ├── utils/               # Utility functions
│   └── visualization/       # Visualization modules
└── tests/                   # Test scripts
```

## Key Files

- `main.py`: Main application entry point
- `run.py`: Entry point that applies patches before running main.py
- `terminal.py`: Terminal UI entry point
- `run_script.py`: Utility to run scripts from the scripts directory
- `README.md`: Main documentation

## Scripts Organization

Utility scripts are organized by category:

- **Download Scripts** (`scripts/download/`): Scripts for downloading LLM models
- **Fix Model Scripts** (`scripts/fix_model/`): Scripts for fixing model downloads
- **Create Scripts** (`scripts/create/`): Scripts for creating distributions and assets
- **Build Scripts** (`scripts/build/`): Scripts for building executables

To run these scripts, use the `run_script.py` utility:

```bash
# List all available scripts
python run_script.py --list

# Run a specific script
python run_script.py --category download --script download_llm_model.py
```

## Documentation

Documentation files are organized in the `docs/` directory:

- `FUNCTION_KEYS.md`: Documentation for function key usage
- `how_to_use.md`: Detailed user guide
- `README_EXECUTABLE.md`: Guide for the executable version
- `README_LLM.md`: Guide for LLM features
- `TERMINAL_UI_GUIDE.md`: Guide for the terminal UI

## Setup

Installation and setup scripts are in the `setup/` directory:

- `install_dependencies.bat`: Script to install dependencies
- `install_keyboard.bat`: Script to install keyboard module
- `install_llm_deps.py`: Script to install LLM dependencies
- `install_pyinstaller.py`: Script to install PyInstaller
