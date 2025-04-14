"""
Direct model download script.

This script downloads LLM models directly using requests without relying on the Hugging Face Hub API.
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

def download_file(url, output_path, chunk_size=8192, use_auth=False):
    """
    Download a file from a URL with progress bar.

    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        chunk_size (int): Size of chunks to download
        use_auth (bool): Whether to use Hugging Face authentication

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare headers for authentication if needed
        headers = {}
        if use_auth and 'huggingface.co' in url:
            # Try to use huggingface_hub for authentication
            try:
                from huggingface_hub import HfApi
                token = HfApi().token
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                    logger.info("Using Hugging Face authentication token")
                    print("Using Hugging Face authentication token")
            except Exception as e:
                logger.warning(f"Could not use Hugging Face authentication: {e}")
                print("Could not use Hugging Face authentication. Trying without authentication...")

        # Make the request
        response = requests.get(url, stream=True, timeout=30, headers=headers)

        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to download file: HTTP status code {response.status_code}")
            if response.status_code == 401 and 'huggingface.co' in url:
                print("Authentication error (401). This model may require Hugging Face login.")
                print("Please run 'huggingface-cli login' to authenticate or try a different model.")
            return False

        # Get the total size
        total_size = int(response.headers.get('content-length', 0))

        # Check if content length is reasonable
        if total_size < 1000000:  # Less than 1MB is suspicious for a model file
            logger.error(f"Content length is suspiciously small: {total_size} bytes")
            return False

        # Download with progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)

        # Check if download was successful
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully downloaded file to {output_path}")
            return True
        else:
            logger.error(f"Downloaded file is empty or does not exist: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def download_model(model_key, force=False):
    """
    Download a model file directly.

    Args:
        model_key (str): Key of the model to download
        force (bool): Whether to force download even if the file exists

    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        print(f"Error: Unknown model '{model_key}'")
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

    # Download the file
    logger.info(f"Downloading model {model_name} ({model_size_gb} GB) from {model_url}")
    print(f"Downloading {model_name} ({model_size_gb} GB)...")
    print(f"This may take a while depending on your internet connection.")

    # Use a temporary file for downloading
    temp_path = f"{file_path}.download"

    # Download the file with authentication
    success = download_file(model_url, temp_path, use_auth=True)

    if success:
        # Rename temp file to final file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing existing file: {e}")
                print(f"Error removing existing file: {e}")
                return False

        try:
            os.rename(temp_path, file_path)
            logger.info(f"Successfully downloaded model {model_name} to {file_path}")
            print(f"Successfully downloaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error renaming temporary file: {e}")
            print(f"Error renaming temporary file: {e}")
            return False
    else:
        logger.error(f"Failed to download model {model_name}")
        print(f"Failed to download model {model_name}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Direct model download script")
    parser.add_argument("--model", required=True, help="Model to download")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("Direct Model Downloader")
    print("=" * 80)

    # Print available models with more details
    print("\nAvailable models:")
    for key, info in AVAILABLE_MODELS.items():
        description = info.get('description', '')
        size_gb = info.get('size_gb', 0)
        trading_focus = info.get('trading_focus', 'Unknown')
        print(f"- {key}: {description} ({size_gb:.2f} GB)")
        print(f"  Trading focus: {trading_focus}")

    # Print model directory
    print(f"\nModel directory: {DEFAULT_MODEL_PATH}")

    # Check if directory exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Creating model directory: {DEFAULT_MODEL_PATH}")
        os.makedirs(DEFAULT_MODEL_PATH, exist_ok=True)

    # Download model
    if args.model not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{args.model}'")
        return 1

    print(f"\nDownloading model: {args.model}")
    success = download_model(args.model, args.force)

    if success:
        print(f"\nSuccessfully downloaded model {args.model}")
        return 0
    else:
        print(f"\nFailed to download model {args.model}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
