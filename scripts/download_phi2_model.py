"""
Script to download the Phi-2 model.

This script downloads the Phi-2 model, which is a small but capable model
that works well for basic analysis and runs on modest hardware.
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

def download_phi2_model(force=False):
    """
    Download the Phi-2 model.
    
    Args:
        force (bool): Whether to force download even if the file exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get model info
    model_key = "phi2"
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Model {model_key} not found in AVAILABLE_MODELS")
        return False
    
    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info.get('name')
    model_url = model_info.get('url')
    model_size_gb = model_info.get('size_gb')
    
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
    parser = argparse.ArgumentParser(description="Download Phi-2 model")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Phi-2 Model Downloader")
    print("=" * 80)
    
    # Print model info
    model_info = AVAILABLE_MODELS.get("phi2", {})
    print(f"\nModel: {model_info.get('description', 'Phi-2')}")
    print(f"Size: {model_info.get('size_gb', 1.35)} GB")
    print(f"Details: {model_info.get('details', 'Small but capable model')}")
    
    # Print model directory
    print(f"\nModel directory: {DEFAULT_MODEL_PATH}")
    
    # Download model
    success = download_phi2_model(args.force)
    
    if success:
        print("\nSuccessfully downloaded Phi-2 model.")
        print("\nYou can now use the model for LLM analysis in the application.")
        return 0
    else:
        print("\nFailed to download Phi-2 model.")
        print("\nPlease try again or choose a different model.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
