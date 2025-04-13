#!/usr/bin/env python
"""
Script to install required dependencies for LLM functionality in 3lacks Scanner.
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies for LLM functionality."""
    print("Installing required dependencies for LLM functionality...")
    
    # List of required packages
    packages = [
        "llama-cpp-python",
        "huggingface-hub",
        "requests",
        "tqdm"
    ]
    
    # Install each package
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            return False
    
    print("\nAll dependencies installed successfully!")
    return True

if __name__ == "__main__":
    install_dependencies()
