# 3lacks Scanner LLM Setup Guide

This guide will help you set up the LLM (Large Language Model) functionality in 3lacks Scanner.

## Prerequisites

- Python 3.8 or higher
- At least 8GB of RAM
- At least 5GB of free disk space

## Installation

1. Run the `install_dependencies.bat` file to install the required dependencies:
   ```
   install_dependencies.bat
   ```

2. If you encounter any issues with model downloads, you can manually download the models from HuggingFace:
   - Phi-3 Mini: https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf
   - Llama 3 8B: https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf

3. Place the downloaded models in the following directory:
   ```
   C:\Users\<YourUsername>\.cache\futures_scanner\models\
   ```

## Using LLM Analysis

1. Start the application:
   ```
   python terminal.py
   ```

2. Go to Settings (option 8)
3. Enable LLM Analysis (option 8)
4. Select an LLM model (option 9)
5. Return to the main menu and select LLM Analysis (option 7)

## Troubleshooting

If you encounter any issues:

1. Check that all dependencies are installed:
   ```
   pip install llama-cpp-python huggingface-hub requests tqdm
   ```

2. Verify that the model files exist and are not corrupted:
   ```
   dir %USERPROFILE%\.cache\futures_scanner\models
   ```

3. Check the log file for detailed error messages:
   ```
   type crypto_scanner.log
   ```

## GPU Acceleration

For faster inference, you can enable GPU acceleration in the Settings menu (option g).
This requires a compatible GPU with CUDA support.
