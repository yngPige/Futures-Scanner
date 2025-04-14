#!/usr/bin/env python
"""
Script to provide instructions for manually downloading the LLM model.

This script provides instructions for manually downloading the LLM model
used by the 3lacks Scanner application.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_instructions():
    """Print instructions for manually downloading the LLM model."""
    # Set model info
    model_url = "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf"
    filename = "llama-3-8b-instruct.Q4_K_M.gguf"
    expected_size_gb = 4.37  # Expected size in GB
    
    # Set output directory
    output_dir = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
    output_path = os.path.join(output_dir, filename)
    
    # Print instructions
    print("\n" + "=" * 80)
    print("Instructions for Manually Downloading the LLM Model")
    print("=" * 80)
    print("\nThe 3lacks Scanner application requires an LLM model to perform analysis.")
    print("Follow these steps to manually download the model:")
    print("\n1. Create the following directory if it doesn't exist:")
    print(f"   {output_dir}")
    print("\n2. Download the model from the following URL:")
    print(f"   {model_url}")
    print("\n3. Save the downloaded file to the following location:")
    print(f"   {output_path}")
    print("\n4. Verify that the file size is approximately 4.37 GB.")
    print("\n5. Once the model is downloaded, you can use the LLM analysis feature in 3lacks Scanner.")
    print("\nNote: The model file is large and may take some time to download depending on your internet connection.")
    print("=" * 80)
    
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory created: {output_dir}")
    except Exception as e:
        print(f"\nError creating output directory: {e}")

def main():
    """Main function."""
    print_instructions()

if __name__ == "__main__":
    main()
