"""
Script to download the Llama 3 8B model.

This script downloads the Llama 3 8B model, which is a powerful model
with good reasoning capabilities for technical analysis.
"""

import os
import sys
import logging
import requests
import argparse
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the model information
from src.analysis.local_llm import AVAILABLE_MODELS, DEFAULT_MODEL_PATH

def download_llama3_model(force=False):
    """
    Download the Llama 3 8B model.
    
    Args:
        force (bool): Whether to force download even if the file exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Define model info manually to ensure we have the correct URL
    model_name = "llama-3-8b-instruct.Q4_K_M.gguf"
    model_url = "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf"
    model_size_gb = 4.37
    
    # Set file path
    file_path = os.path.join(DEFAULT_MODEL_PATH, model_name)
    
    # Check if file exists
    if os.path.exists(file_path) and not force:
        file_size = os.path.getsize(file_path)
        expected_size_bytes = model_size_gb * 1024 * 1024 * 1024
        
        # File should be at least 90% of expected size
        if file_size > (expected_size_bytes * 0.9):
            logger.info(f"Model file {model_name} already exists and is valid.")
            print(f"Model file {model_name} already exists and is valid.")
            return True
        else:
            logger.warning(f"Model file {model_name} exists but is too small: {file_size} bytes (expected ~{expected_size_bytes} bytes)")
            print(f"Model file {model_name} exists but is too small. Re-downloading...")
            
            # Remove the existing file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing existing file: {e}")
                print(f"Error removing existing file: {e}")
                return False
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Download the file
    logger.info(f"Downloading model {model_name} ({model_size_gb} GB) from {model_url}")
    print(f"Downloading {model_name} ({model_size_gb} GB)...")
    print(f"This may take a while depending on your internet connection.")
    
    try:
        # Make the request
        response = requests.get(model_url, stream=True, timeout=30)
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP status code {response.status_code}")
            print(f"Failed to download model: HTTP status code {response.status_code}")
            return False
        
        # Get the total size
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if content length is reasonable
        if total_size < 1000000:  # Less than 1MB is suspicious for a model file
            logger.error(f"Content length is suspiciously small: {total_size} bytes")
            print(f"Content length is suspiciously small: {total_size} bytes")
            return False
        
        # Download with progress bar
        with open(file_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        # Check if download was successful
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Successfully downloaded model {model_name} to {file_path}")
            print(f"Successfully downloaded model {model_name}")
            return True
        else:
            logger.error(f"Downloaded file is empty or does not exist: {file_path}")
            print(f"Error: Downloaded file is empty or does not exist")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        print(f"Error downloading model: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Llama 3 8B model")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Llama 3 8B Model Downloader")
    print("=" * 80)
    
    # Print model info
    print(f"\nModel: Llama 3 8B Instruct (Quantized 4-bit)")
    print(f"Size: 4.37 GB")
    print(f"Details: General-purpose LLM with good performance across various tasks")
    
    # Print model directory
    print(f"\nModel directory: {DEFAULT_MODEL_PATH}")
    
    # Download model
    success = download_llama3_model(args.force)
    
    if success:
        print("\nSuccessfully downloaded Llama 3 8B model.")
        print("\nYou can now use the model for LLM analysis in the application.")
        return 0
    else:
        print("\nFailed to download Llama 3 8B model.")
        print("\nPlease try again or choose a different model.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
