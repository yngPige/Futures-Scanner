#!/usr/bin/env python
"""
Script to fix the LLM model download for 3lacks Scanner using curl.
"""
import os
import sys
import subprocess

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
    
    # Download model using curl
    print(f"Downloading model from {model['url']}...")
    try:
        # Use curl to download the file
        curl_command = [
            "curl", "-L", model['url'], 
            "--output", destination,
            "--create-dirs",
            "--progress-bar"
        ]
        
        print(f"Running command: {' '.join(curl_command)}")
        subprocess.run(curl_command, check=True)
        
        # Verify the downloaded file size
        if os.path.exists(destination):
            file_size_mb = os.path.getsize(destination) / (1024 * 1024)
            expected_size_mb = model['size_gb'] * 1024  # Convert GB to MB
            
            if file_size_mb < (expected_size_mb * 0.9):
                print(f"Downloaded file is too small: {file_size_mb:.2f} MB (expected ~{expected_size_mb:.2f} MB)")
            else:
                print(f"Model downloaded successfully to {destination} with size {file_size_mb:.2f} MB")
    
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
