"""
Simple script to build the 3lacks Scanner executable.
"""

import os
import sys
import subprocess

def build_executable():
    """Build the 3lacks Scanner executable."""
    print("Building 3lacks Scanner executable...")

    # Create necessary directories
    for directory in ['models', 'data', 'results', 'charts']:
        os.makedirs(directory, exist_ok=True)

    # Run PyInstaller with enhanced options
    subprocess.check_call([
        sys.executable,
        "-m",
        "PyInstaller",
        "terminal.py",
        "--name=3lacks_Scanner",
        "--onefile",
        "--clean",
        "--additional-hooks-dir=.",
        "--hidden-import=typing_extensions",
        "--hidden-import=importlib_metadata",
        "--collect-metadata=typing_extensions",
        "--collect-metadata=pandas",
        "--collect-metadata=numpy",
        "--collect-metadata=scikit-learn",
        "--collect-metadata=matplotlib",
        "--collect-metadata=plotly",
        "--runtime-hook=rth_typing_extensions.py"
    ])

    print("Executable built successfully.")

if __name__ == "__main__":
    build_executable()
