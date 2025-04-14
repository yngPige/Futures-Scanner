#!/usr/bin/env python
"""
Script to download and verify LLM models for 3lacks Scanner.

This script provides a command-line interface to download and verify LLM models
used by the 3lacks Scanner application.
"""

import os
import sys
import logging
import argparse
import requests
from tqdm import tqdm
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local_llm module
try:
    from src.analysis.local_llm import AVAILABLE_MODELS, DEFAULT_MODEL_PATH
except ImportError:
    # If running from the root directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from src.analysis.local_llm import AVAILABLE_MODELS, DEFAULT_MODEL_PATH
    except ImportError:
        logger.error("Failed to import local_llm module. Make sure you're running this script from the project root directory.")
        sys.exit(1)

def calculate_md5(file_path):
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(model_key, output_dir=None, force=False):
    """
    Download an LLM model.
    
    Args:
        model_key (str): Key of the model to download
        output_dir (str, optional): Directory to save the model. If None, uses default.
        force (bool): Whether to force download even if the model already exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info.get('name')
    model_url = model_info.get('url')
    model_size = model_info.get('size_gb', 'unknown')
    
    if not model_name or not model_url:
        logger.error(f"Invalid model info for {model_key}: missing name or URL")
        return False
    
    # Set output directory
    output_dir = output_dir or DEFAULT_MODEL_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file path
    output_path = os.path.join(output_dir, model_name)
    temp_path = f"{output_path}.download"
    
    # Check if model already exists
    if os.path.exists(output_path) and not force:
        file_size_gb = os.path.getsize(output_path) / (1024 * 1024 * 1024)
        logger.info(f"Model already exists at {output_path} ({file_size_gb:.2f} GB)")
        
        # Verify file size
        if isinstance(model_size, (int, float)):
            expected_size_bytes = model_size * 1024 * 1024 * 1024
            actual_size = os.path.getsize(output_path)
            size_diff_percent = abs(actual_size - expected_size_bytes) / expected_size_bytes * 100
            
            if size_diff_percent > 20:
                logger.warning(f"File size ({file_size_gb:.2f} GB) differs significantly from expected size ({model_size} GB)")
                if not force:
                    logger.info("Use --force to re-download the model")
                    return True
        
        return True
    
    # Download the model
    logger.info(f"Downloading {model_name} ({model_size} GB) from {model_url}")
    
    try:
        # Make the request
        response = requests.get(model_url, stream=True, timeout=30)
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP status code {response.status_code}")
            return False
        
        # Get the total size
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if content length is reasonable
        if total_size < 1000000:  # Less than 1MB is suspicious for a model file
            logger.error(f"Content length is suspiciously small: {total_size} bytes")
            return False
        
        # Download with progress bar
        with open(temp_path, 'wb') as f, tqdm(
            desc=model_name,
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
            if isinstance(model_size, (int, float)):
                expected_size_bytes = model_size * 1024 * 1024 * 1024
                actual_size = os.path.getsize(temp_path)
                size_diff_percent = abs(actual_size - expected_size_bytes) / expected_size_bytes * 100
                
                if size_diff_percent > 20:
                    logger.warning(f"Downloaded file size ({actual_size / (1024*1024*1024):.2f} GB) differs significantly from expected size ({model_size} GB)")
            
            # Move the temporary file to the final location
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                logger.info(f"Successfully downloaded model to {output_path}")
                
                # Calculate and display MD5 hash
                md5_hash = calculate_md5(output_path)
                logger.info(f"MD5 hash: {md5_hash}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to move temporary file to final location: {e}")
                return False
        else:
            logger.error(f"Downloaded file is empty or does not exist: {temp_path}")
            return False
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during download: {e}")
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

def list_models():
    """List all available models."""
    logger.info("Available models:")
    
    for key, info in AVAILABLE_MODELS.items():
        name = info.get('name', 'Unknown')
        size = info.get('size_gb', 'Unknown')
        description = info.get('description', '')
        trading_focus = info.get('trading_focus', 'Unknown')
        
        print(f"- {key}:")
        print(f"  Name: {name}")
        print(f"  Size: {size} GB")
        print(f"  Description: {description}")
        print(f"  Trading Focus: {trading_focus}")
        print()

def verify_model(model_key, model_dir=None):
    """
    Verify that a model exists and has the expected size.
    
    Args:
        model_key (str): Key of the model to verify
        model_dir (str, optional): Directory where the model is stored. If None, uses default.
        
    Returns:
        bool: True if the model exists and has the expected size, False otherwise
    """
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    model_info = AVAILABLE_MODELS[model_key]
    model_name = model_info.get('name')
    model_size = model_info.get('size_gb')
    
    if not model_name:
        logger.error(f"Invalid model info for {model_key}: missing name")
        return False
    
    # Set model directory
    model_dir = model_dir or DEFAULT_MODEL_PATH
    
    # Set model file path
    model_path = os.path.join(model_dir, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    logger.info(f"Model file size: {file_size_gb:.2f} GB")
    
    # Verify file size if expected size is provided
    if isinstance(model_size, (int, float)):
        expected_size_bytes = model_size * 1024 * 1024 * 1024
        size_diff_percent = abs(file_size - expected_size_bytes) / expected_size_bytes * 100
        
        if size_diff_percent > 20:
            logger.warning(f"File size ({file_size_gb:.2f} GB) differs significantly from expected size ({model_size} GB)")
            return False
    
    # Calculate and display MD5 hash
    md5_hash = calculate_md5(model_path)
    logger.info(f"MD5 hash: {md5_hash}")
    
    logger.info(f"Model verification successful: {model_path}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download and verify LLM models for 3lacks Scanner")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Model key to download")
    download_parser.add_argument("--output-dir", help="Directory to save the model")
    download_parser.add_argument("--force", action="store_true", help="Force download even if the model already exists")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a model")
    verify_parser.add_argument("model", help="Model key to verify")
    verify_parser.add_argument("--model-dir", help="Directory where the model is stored")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "download":
        success = download_model(args.model, args.output_dir, args.force)
        if not success:
            sys.exit(1)
    elif args.command == "list":
        list_models()
    elif args.command == "verify":
        success = verify_model(args.model, args.model_dir)
        if not success:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
