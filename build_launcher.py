#!/usr/bin/env python
"""
Build a Windows executable launcher for Blacks Scanner.

This script builds a Windows executable (.exe) file that launches the Blacks Scanner
in a new console window.
"""

import os
import sys
import subprocess

def build_launcher():
    """Build a Windows executable launcher for Blacks Scanner."""
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Get the path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the path to the PyBloat directory
    pybloat_dir = os.path.join(current_dir, 'Py Bloat')

    # Get the path to the icon file
    icon_path = os.path.join(pybloat_dir, 'icon.ico')

    # Build the executable
    print("Building launcher executable...")
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--console",
        f"--icon={icon_path}" if os.path.exists(icon_path) else "",
        "--name=Blacks_Scanner",
        "launch_scanner_gui.py"
    ])

    print("Launcher executable built successfully.")
    print(f"Executable location: {os.path.join(current_dir, 'dist', 'Blacks_Scanner.exe')}")

if __name__ == "__main__":
    try:
        build_launcher()
    except Exception as e:
        print(f"Error building launcher: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
