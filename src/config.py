"""
Configuration module for 3lacks Scanner.

This module provides configuration settings for the application.
"""

import os

# Path to the PyBloat directory (where executables are stored)
PYBLOAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Py Bloat')

# Ensure the PyBloat directory exists
if not os.path.exists(PYBLOAT_DIR):
    raise FileNotFoundError(f"PyBloat directory not found at {PYBLOAT_DIR}")

# Function to get the path to a file in the PyBloat directory
def get_pybloat_path(filename):
    """
    Get the path to a file in the PyBloat directory.
    
    Args:
        filename (str): The name of the file
        
    Returns:
        str: The full path to the file in the PyBloat directory
    """
    path = os.path.join(PYBLOAT_DIR, filename)
    return path

# Function to check if a file exists in the PyBloat directory
def pybloat_file_exists(filename):
    """
    Check if a file exists in the PyBloat directory.
    
    Args:
        filename (str): The name of the file
        
    Returns:
        bool: True if the file exists, False otherwise
    """
    path = get_pybloat_path(filename)
    return os.path.exists(path)
