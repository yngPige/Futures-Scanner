#!/usr/bin/env python
"""
Launcher script for 3lacks Scanner.

This script launches the 3lacks Scanner application using the PyBloat directory.
"""

import os
import sys
import subprocess

def main():
    """Launch the 3lacks Scanner application."""
    # Get the path to the PyBloat directory
    pybloat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Py Bloat')
    
    # Check if the PyBloat directory exists
    if not os.path.exists(pybloat_dir):
        print(f"Error: PyBloat directory not found at {pybloat_dir}")
        input("Press Enter to exit...")
        return
    
    # Change to the PyBloat directory
    os.chdir(pybloat_dir)
    
    # Launch the terminal.py script
    terminal_path = os.path.join(pybloat_dir, 'terminal.py')
    if os.path.exists(terminal_path):
        print(f"Launching 3lacks Scanner from {pybloat_dir}...")
        subprocess.run([sys.executable, terminal_path])
    else:
        print(f"Error: terminal.py not found in {pybloat_dir}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
