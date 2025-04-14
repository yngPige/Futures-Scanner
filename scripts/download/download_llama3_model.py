#!/usr/bin/env python
"""
Script to download the Llama 3 model for 3lacks Scanner.

This script downloads the Llama 3 model using the Hugging Face Hub API.
"""

import os
import sys
import logging
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_llama3_model():
    """Download the Llama 3 model."""
    # Set model info
    model_url = "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf"
    filename = "llama-3-8b-instruct.Q4_K_M.gguf"
    expected_size_gb = 4.37  # Expected size in GB

    # Set output directory
    output_dir = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
    os.makedirs(output_dir, exist_ok=True)

    # Set output file path
    output_path = os.path.join(output_dir, filename)
    temp_path = f"{output_path}.download"

    # Check if model already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        file_size_gb = os.path.getsize(output_path) / (1024 * 1024 * 1024)
        logger.info(f"Model already exists at {output_path} ({file_size_gb:.2f} GB)")
        return True

    # Download the model
    logger.info(f"Downloading model from URL: {model_url}")

    try:
        # Download with progress bar
        response = requests.get(model_url, stream=True, timeout=30)

        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP status code {response.status_code}")
            return False

        total_size = int(response.headers.get('content-length', 0))

        # Check if content length is reasonable
        if total_size < 1000000:  # Less than 1MB is suspicious for a model file
            logger.error(f"Content length is suspiciously small: {total_size} bytes")
            return False

        # Download with progress bar
        with open(temp_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)

        # Verify the file was downloaded correctly
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            # Check if the file size is reasonable
            actual_size = os.path.getsize(temp_path)
            actual_size_gb = actual_size / (1024 * 1024 * 1024)
            expected_size_bytes = expected_size_gb * 1024 * 1024 * 1024
            size_diff_percent = abs(actual_size - expected_size_bytes) / expected_size_bytes * 100

            if size_diff_percent > 20:
                logger.warning(f"Downloaded file size ({actual_size_gb:.2f} GB) differs significantly from expected size ({expected_size_gb} GB)")

            # Move the temporary file to the final location
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                logger.info(f"Successfully downloaded model to {output_path} ({actual_size_gb:.2f} GB)")
                return True
            except Exception as e:
                logger.error(f"Failed to move temporary file to final location: {e}")
                return False
        else:
            logger.error(f"Downloaded file is empty or does not exist: {temp_path}")
            return False

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False
    finally:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

def main():
    """Main function."""
    print("Downloading Llama 3 model for 3lacks Scanner...")

    try:
        success = download_llama3_model()

        if success:
            print("\nModel downloaded successfully!")
            print("You can now use the LLM analysis feature in 3lacks Scanner.")
        else:
            print("\nFailed to download model.")
            print("Please try again later or download the model manually.")
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
