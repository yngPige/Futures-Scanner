"""
Script to install PyInstaller.
"""

import subprocess
import sys

def main():
    """Install PyInstaller."""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    print("PyInstaller installed successfully.")

if __name__ == "__main__":
    main()
