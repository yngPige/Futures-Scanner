#!/usr/bin/env python
"""
Script to download the Phi-3 Mini model for 3lacks Scanner.
"""
import os
import sys
import requests
from tqdm import tqdm

# Define model URL and destination
MODEL_URL = "https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf"
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")
DESTINATION = os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct.Q4_K_M.gguf")

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
        print(f"Downloading Phi-3 Mini model to {destination}...")
        print(f"This may take a while. Please be patient.")
        
        # Use a session with custom headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        response = session.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if the response is valid
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Download the file
        with open(destination, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)
        
        # Verify the file size
        file_size = os.path.getsize(destination)
        if file_size < 1000000:  # Less than 1MB
            print(f"Warning: Downloaded file is too small ({file_size} bytes)")
            return False
        
        print(f"Successfully downloaded to {destination}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        return True
    
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

if __name__ == "__main__":
    # Download the model
    success = download_file(MODEL_URL, DESTINATION)
    
    if success:
        print("Download completed successfully!")
    else:
        print("Download failed. Please try again or download manually.")
        print(f"Manual download URL: {MODEL_URL}")
        print(f"Save to: {DESTINATION}")
    
    input("Press Enter to exit...")
