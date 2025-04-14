"""
Script to build the 3lacks Scanner executable.
"""

import os
import sys
import subprocess
import shutil
import zipfile
from datetime import datetime

def build_executable():
    """Build the 3lacks Scanner executable."""
    print("Building 3lacks Scanner executable...")

    # Create necessary directories
    for directory in ['models', 'data', 'results', 'charts']:
        os.makedirs(directory, exist_ok=True)

    # Run PyInstaller
    subprocess.check_call([
        sys.executable,
        "-m",
        "PyInstaller",
        "3lacks_scanner.spec",
        "--clean"
    ])

    print("Executable built successfully.")

    # Create a ZIP file for distribution
    create_distribution_package()

def create_distribution_package():
    """Create a ZIP file for distribution."""
    print("Creating distribution package...")

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create ZIP file
    zip_filename = f"3lacks_Scanner_{timestamp}.zip"

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the dist directory
        for root, _, files in os.walk("dist/3lacks_Scanner"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "dist")
                zipf.write(file_path, arcname)

        # Add README and other documentation
        for file in ["README_EXECUTABLE.md", "README.md", "FUNCTION_KEYS.md"]:
            if os.path.exists(file):
                # If it's the executable README, rename it to README.md in the zip
                if file == "README_EXECUTABLE.md":
                    zipf.write(file, "README.md")
                else:
                    zipf.write(file)

    print(f"Distribution package created: {zip_filename}")

if __name__ == "__main__":
    build_executable()
