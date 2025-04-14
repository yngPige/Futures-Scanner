# 3lacks Scanner Utilities

This directory contains utility functions for the 3lacks Scanner application, organized by category.

## Directory Structure

```
utils/
├── build/              # Build utilities
│   └── hooks/          # PyInstaller hooks
├── data/               # Data utilities
├── docs/               # Documentation
├── models/             # Model utilities
├── setup/              # Setup utilities
└── visualization/      # Visualization utilities
```

## Running Utilities

You can run any utility using the `run_utils.py` script:

```bash
# List all available utilities
python run_utils.py --list

# Run a specific utility
python run_utils.py --category data --module download_llm_model.py
```

## Utility Categories

### Data Utilities

Utilities for downloading and managing data:

- `download_llm_model.py`: Download LLM models
- `download_llama3_model.py`: Download Llama3 models
- `download_model.py`: General model downloader
- `download_phi3_mini.py`: Download Phi3 Mini model

### Model Utilities

Utilities for fixing and managing models:

- `fix_model_download.py`: Fix general model downloads
- `fix_model_download_curl.py`: Fix downloads using curl
- `fix_model_download_hf.py`: Fix HuggingFace downloads
- `fix_model_download_phi3.py`: Fix Phi3 model downloads

### Build Utilities

Utilities for building executables:

- `build_executable.py`: Build the 3lacks Scanner executable
- `simple_build.py`: Simple build script
- `fix_build.py`: Fix build issues

### Setup Utilities

Utilities for setting up the application:

- `install_llm_deps.py`: Install LLM dependencies
- `install_pyinstaller.py`: Install PyInstaller

### Visualization Utilities

Utilities for creating visualizations:

- `create_distribution.py`: Create a distribution package
- `create_fixed_distribution.py`: Create a fixed distribution package
- `create_icon.py`: Create application icons
