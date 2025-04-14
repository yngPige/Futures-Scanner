#!/usr/bin/env python
"""
Script to fix the LLM model download for 3lacks Scanner.
"""
import os
import sys
import requests
from tqdm import tqdm
import shutil

def download_file(url, destination):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"Downloading {os.path.basename(destination)} ({total_size / (1024*1024*1024):.2f} GB)...")
        
        # Use a temporary file for downloading
        temp_destination = destination + ".tmp"
        
        with open(temp_destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)
        
        # Verify the file was downloaded correctly
        if os.path.exists(temp_destination) and os.path.getsize(temp_destination) > 0:
            # Move the temporary file to the final destination
            shutil.move(temp_destination, destination)
            print(f"Successfully downloaded to {destination}")
            return True
        else:
            print(f"Downloaded file is empty or does not exist: {temp_destination}")
            return False
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    """Main function to download models."""
    # Define models
    models = {
        "llama3-8b": {
            "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
            "filename": "llama-3-8b-instruct.Q4_K_M.gguf",
            "size_gb": 4.37
        }
    }
    
    # Define models directory
    models_dir = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model file exists
    model = models["llama3-8b"]
    destination = os.path.join(models_dir, model['filename'])
    
    if os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        expected_size_mb = model['size_gb'] * 1024  # Convert GB to MB
        
        if file_size_mb < (expected_size_mb * 0.9):
            print(f"Found incomplete model file at {destination}")
            print(f"File size: {file_size_mb:.2f} MB is too small (expected ~{expected_size_mb:.2f} MB)")
            print(f"Removing incomplete file and downloading again...")
            os.remove(destination)
        else:
            print(f"Model already exists at {destination} with size {file_size_mb:.2f} MB")
            return
    
    # Download model
    download_file(model['url'], destination)
    
    # Verify the downloaded file size
    if os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        expected_size_mb = model['size_gb'] * 1024  # Convert GB to MB
        
        if file_size_mb < (expected_size_mb * 0.9):
            print(f"Downloaded file is too small: {file_size_mb:.2f} MB (expected ~{expected_size_mb:.2f} MB)")
        else:
            print(f"Model downloaded successfully with size {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
