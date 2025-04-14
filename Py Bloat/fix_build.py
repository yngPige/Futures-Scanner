"""
Script to build the 3lacks Scanner executable with fixes for typing_extensions.
"""

import os
import sys
import subprocess
import shutil

def build_executable():
    """Build the 3lacks Scanner executable with fixes."""
    print("Building 3lacks Scanner executable with fixes...")
    
    # Create necessary directories
    for directory in ['models', 'data', 'results', 'charts']:
        os.makedirs(directory, exist_ok=True)
    
    # Create a temporary directory for the build
    build_dir = "build_temp"
    os.makedirs(build_dir, exist_ok=True)
    
    # Copy typing_extensions metadata to the build directory
    try:
        import importlib.metadata
        import typing_extensions
        
        # Get the path to typing_extensions
        typing_extensions_path = os.path.dirname(typing_extensions.__file__)
        print(f"typing_extensions path: {typing_extensions_path}")
        
        # Create the metadata directory structure
        metadata_dir = os.path.join(build_dir, "typing_extensions-metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create a simple metadata file
        with open(os.path.join(metadata_dir, "METADATA"), "w") as f:
            f.write("Name: typing-extensions\nVersion: 4.0.0\n")
        
        print("Created typing_extensions metadata")
    except (ImportError, Exception) as e:
        print(f"Error creating typing_extensions metadata: {e}")
    
    # Run PyInstaller with basic options
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "PyInstaller",
        "--name=3lacks_Scanner_Fixed",
        "--onefile",
        "--clean",
        "--add-data", f"{build_dir}/typing_extensions-metadata;typing_extensions-metadata",
        "--hidden-import=typing_extensions",
        "--hidden-import=importlib.metadata",
        "terminal.py"
    ])
    
    print("Executable built successfully.")
    
    # Clean up
    shutil.rmtree(build_dir)

if __name__ == "__main__":
    build_executable()
