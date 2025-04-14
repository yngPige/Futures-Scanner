# 3lacks Scanner Scripts

This directory contains utility scripts for the 3lacks Scanner application, organized by category.

## Directory Structure

- `download/`: Scripts for downloading LLM models
- `fix_model/`: Scripts for fixing model downloads
- `create/`: Scripts for creating distributions and assets
- `build/`: Scripts for building executables

## Running Scripts

You can run any script using the `run_script.py` utility:

```bash
# List all available scripts
python run_script.py --list

# Run a specific script
python run_script.py --category download --script download_llm_model.py
```

## Script Categories

### Download Scripts

Scripts for downloading LLM models:

- `download_llm_model.py`: Download and verify LLM models
- `download_llama3_model.py`: Download Llama3 models
- `download_model.py`: General model downloader
- `download_phi3_mini.py`: Download Phi3 Mini model

### Fix Model Scripts

Scripts for fixing model downloads:

- `fix_model_download.py`: Fix general model downloads
- `fix_model_download_curl.py`: Fix downloads using curl
- `fix_model_download_hf.py`: Fix HuggingFace downloads
- `fix_model_download_phi3.py`: Fix Phi3 model downloads

### Create Scripts

Scripts for creating distributions and assets:

- `create_distribution.py`: Create a distribution package
- `create_fixed_distribution.py`: Create a fixed distribution package
- `create_icon.py`: Create application icons

### Build Scripts

Scripts for building executables:

- `build_executable.py`: Build the 3lacks Scanner executable
