"""
Cleanup script for LLM models.

This script removes corrupted model files and updates the model list.
"""

import os
import sys
import shutil
import logging
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the logging utility
from src.utils.logging_utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Import the model information
from src.analysis.local_llm import DEFAULT_MODEL_PATH

def cleanup_models_directory(force=False):
    """
    Clean up the models directory by removing all files.
    
    Args:
        force (bool): Whether to force removal without confirmation
        
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    if not os.path.exists(DEFAULT_MODEL_PATH):
        logger.info(f"Models directory does not exist: {DEFAULT_MODEL_PATH}")
        print(f"Models directory does not exist: {DEFAULT_MODEL_PATH}")
        return True
    
    # List all files in the directory
    files = [f for f in os.listdir(DEFAULT_MODEL_PATH) if os.path.isfile(os.path.join(DEFAULT_MODEL_PATH, f))]
    
    if not files:
        logger.info(f"No files found in models directory: {DEFAULT_MODEL_PATH}")
        print(f"No files found in models directory: {DEFAULT_MODEL_PATH}")
        return True
    
    # Print files to be removed
    print(f"Found {len(files)} files in models directory:")
    for file in files:
        file_path = os.path.join(DEFAULT_MODEL_PATH, file)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb < 10:  # Less than 10MB is likely corrupted
            status = "CORRUPTED"
        else:
            status = "OK"
        
        print(f"- {file} ({file_size_mb:.2f} MB) [{status}]")
    
    # Confirm removal
    if not force:
        confirm = input("\nAre you sure you want to remove all these files? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return False
    
    # Remove all files
    print("\nRemoving files...")
    success = True
    
    for file in files:
        file_path = os.path.join(DEFAULT_MODEL_PATH, file)
        try:
            os.remove(file_path)
            print(f"Removed: {file}")
        except Exception as e:
            logger.error(f"Error removing file {file}: {e}")
            print(f"Error removing file {file}: {e}")
            success = False
    
    # Check if any subdirectories exist and remove them
    subdirs = [d for d in os.listdir(DEFAULT_MODEL_PATH) if os.path.isdir(os.path.join(DEFAULT_MODEL_PATH, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(DEFAULT_MODEL_PATH, subdir)
        try:
            shutil.rmtree(subdir_path)
            print(f"Removed directory: {subdir}")
        except Exception as e:
            logger.error(f"Error removing directory {subdir}: {e}")
            print(f"Error removing directory {subdir}: {e}")
            success = False
    
    return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cleanup script for LLM models")
    parser.add_argument("--force", action="store_true", help="Force removal without confirmation")
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("LLM Models Cleanup")
    print("=" * 80)
    
    # Print model directory
    print(f"\nModel directory: {DEFAULT_MODEL_PATH}")
    
    # Clean up models directory
    success = cleanup_models_directory(args.force)
    
    if success:
        print("\nSuccessfully cleaned up models directory.")
        
        # Print next steps
        print("\nNext steps:")
        print("1. Update the model list in src/analysis/local_llm.py to include only models that work.")
        print("2. Download a working model using the direct_model_download.py script.")
        
        return 0
    else:
        print("\nFailed to clean up models directory.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
