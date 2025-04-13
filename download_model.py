#!/usr/bin/env python
"""
Script to download LLM models for 3lacks Scanner.
"""
import os
import sys
import requests
from tqdm import tqdm

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
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                size = f.write(data)
                bar.update(size)
        
        print(f"Successfully downloaded to {destination}")
        return True
    
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    """Main function to download models."""
    # Define models
    models = {
        "phi3-mini": {
            "url": "https://huggingface.co/TheBloke/phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf",
            "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
            "size_gb": 1.91
        },
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
    
    # Print available models
    print("Available models:")
    for i, (key, model) in enumerate(models.items(), 1):
        print(f"{i}. {key} - {model['filename']} ({model['size_gb']} GB)")
    
    # Get user input
    try:
        choice = input("\nEnter model number to download (or 'all' to download all models): ")
        
        if choice.lower() == 'all':
            # Download all models
            for key, model in models.items():
                destination = os.path.join(models_dir, model['filename'])
                download_file(model['url'], destination)
        else:
            # Download selected model
            choice = int(choice)
            if choice < 1 or choice > len(models):
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
                return
            
            # Get selected model
            key = list(models.keys())[choice - 1]
            model = models[key]
            
            # Download model
            destination = os.path.join(models_dir, model['filename'])
            download_file(model['url'], destination)
    
    except ValueError:
        print("Invalid input. Please enter a number or 'all'.")
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
