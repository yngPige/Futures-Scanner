"""
Script to fix LLM model downloads.

This script checks for corrupted model files and re-downloads them if necessary.
"""

import os
import sys
import logging
import requests
import argparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the model information
from src.analysis.local_llm import AVAILABLE_MODELS, DEFAULT_MODEL_PATH

def check_model_file(model_key):
    """
    Check if a model file exists and is valid.

    Args:
        model_key (str): Key of the model to check

    Returns:
        tuple: (exists, valid, path) - whether the file exists, is valid, and its path
    """
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False, False, None

    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info.get('name')
    expected_size_gb = model_info.get('size_gb')

    # Set file path
    file_path = os.path.join(DEFAULT_MODEL_PATH, model_name)

    # Check if file exists
    exists = os.path.exists(file_path)

    # Check if file is valid (not empty or too small)
    valid = False
    if exists:
        file_size = os.path.getsize(file_path)
        expected_size_bytes = expected_size_gb * 1024 * 1024 * 1024

        # File should be at least 90% of expected size
        valid = file_size > (expected_size_bytes * 0.9)

        if not valid:
            logger.warning(f"Model file {model_name} is too small: {file_size} bytes (expected ~{expected_size_bytes} bytes)")

    return exists, valid, file_path

def download_model(model_key, force=False):
    """
    Download a model file using Hugging Face Hub API.

    Args:
        model_key (str): Key of the model to download
        force (bool): Whether to force download even if the file exists

    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False

    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info.get('name')
    model_url = model_info.get('url')
    model_size_gb = model_info.get('size_gb')

    # Set file path
    file_path = os.path.join(DEFAULT_MODEL_PATH, model_name)

    # Check if file exists and is valid
    exists, valid, _ = check_model_file(model_key)

    if exists and valid and not force:
        logger.info(f"Model file {model_name} already exists and is valid.")
        return True

    # Create directory if it doesn't exist
    os.makedirs(DEFAULT_MODEL_PATH, exist_ok=True)

    # Parse the URL to get repo_id and filename
    # URL format: https://huggingface.co/{repo_id}/resolve/main/{filename}
    try:
        # Extract repo_id and filename from URL
        url_parts = model_url.split('huggingface.co/')[1].split('/resolve/')
        repo_id = url_parts[0]
        filename = url_parts[1].split('/', 1)[1]

        logger.info(f"Downloading model {model_name} ({model_size_gb} GB) from {repo_id}")
        print(f"Downloading {model_name} ({model_size_gb} GB) from Hugging Face...")
        print(f"This may take a while depending on your internet connection.")

        # Download the file using Hugging Face Hub API
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=DEFAULT_MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=force
        )

        # Rename the file if necessary
        if os.path.basename(downloaded_path) != model_name:
            new_path = os.path.join(DEFAULT_MODEL_PATH, model_name)
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(downloaded_path, new_path)
            downloaded_path = new_path

        # Check if download was successful
        if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 0:
            logger.info(f"Successfully downloaded model {model_name} to {downloaded_path}")
            print(f"Successfully downloaded model {model_name}")
            return True
        else:
            logger.error(f"Downloaded file is empty or does not exist: {downloaded_path}")
            print(f"Error: Downloaded file is empty or does not exist")
            return False

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        print(f"Error downloading model: {e}")
        return False

def fix_all_models(force=False):
    """
    Check and fix all model files.

    Args:
        force (bool): Whether to force download even if the file exists

    Returns:
        bool: True if all models were fixed, False otherwise
    """
    all_success = True

    for model_key in AVAILABLE_MODELS:
        logger.info(f"Checking model {model_key}...")

        exists, valid, file_path = check_model_file(model_key)

        if exists and not valid:
            logger.warning(f"Model file for {model_key} exists but is invalid. Removing...")
            try:
                os.remove(file_path)
                exists = False
            except Exception as e:
                logger.error(f"Error removing invalid model file: {e}")

        if not exists or force:
            logger.info(f"Downloading model {model_key}...")
            success = download_model(model_key, force)

            if not success:
                all_success = False
                logger.error(f"Failed to download model {model_key}")
        else:
            logger.info(f"Model {model_key} is already valid.")

    return all_success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix LLM model downloads")
    parser.add_argument("--model", help="Specific model to fix (default: all models)")
    parser.add_argument("--force", action="store_true", help="Force download even if file exists")
    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("LLM Model Fixer")
    print("=" * 80)

    # Print available models
    print("\nAvailable models:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"- {key}: {info.get('name')} ({info.get('size_gb')} GB)")

    # Print model directory
    print(f"\nModel directory: {DEFAULT_MODEL_PATH}")

    # Check if directory exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Creating model directory: {DEFAULT_MODEL_PATH}")
        os.makedirs(DEFAULT_MODEL_PATH, exist_ok=True)

    # Fix models
    if args.model:
        if args.model not in AVAILABLE_MODELS:
            print(f"Error: Unknown model '{args.model}'")
            return 1

        print(f"\nFixing model: {args.model}")
        exists, valid, file_path = check_model_file(args.model)

        if exists and valid and not args.force:
            print(f"Model {args.model} is already valid.")
            return 0

        success = download_model(args.model, args.force)

        if success:
            print(f"\nSuccessfully fixed model {args.model}")
            return 0
        else:
            print(f"\nFailed to fix model {args.model}")
            return 1
    else:
        print("\nFixing all models...")
        success = fix_all_models(args.force)

        if success:
            print("\nSuccessfully fixed all models")
            return 0
        else:
            print("\nFailed to fix some models")
            return 1

if __name__ == "__main__":
    sys.exit(main())
