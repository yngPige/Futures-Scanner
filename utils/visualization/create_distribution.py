"""
Script to create a distribution package for 3lacks Scanner.
"""

import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import shutil
import zipfile
from datetime import datetime

def create_distribution_package():
    """Create a distribution package for 3lacks Scanner."""
    print("Creating distribution package...")
    
    # Create necessary directories
    for directory in ['models', 'data', 'results', 'charts']:
        os.makedirs(directory, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create distribution directory
    dist_dir = f"3lacks_Scanner_Distribution_{timestamp}"
    os.makedirs(dist_dir, exist_ok=True)
    
    # Copy executable
    shutil.copy("dist/3lacks_Scanner.exe", dist_dir)
    
    # Copy necessary directories
    for directory in ['models', 'data', 'results', 'charts', 'docs']:
        if os.path.exists(directory):
            shutil.copytree(directory, os.path.join(dist_dir, directory))
    
    # Copy necessary files
    for file in ["README_EXECUTABLE.md", "FUNCTION_KEYS.md", "install_keyboard.bat", "install_keyboard.py", 
                "download_llm_model.py", "download_llama3_model.py", "llm_model_instructions.py"]:
        if os.path.exists(file):
            shutil.copy(file, dist_dir)
    
    # Rename README_EXECUTABLE.md to README.md in the distribution directory
    if os.path.exists(os.path.join(dist_dir, "README_EXECUTABLE.md")):
        shutil.move(os.path.join(dist_dir, "README_EXECUTABLE.md"), os.path.join(dist_dir, "README.md"))
    
    # Create ZIP file
    zip_filename = f"3lacks_Scanner_Distribution_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dist_dir)
                zipf.write(file_path, arcname)
    
    print(f"Distribution package created: {zip_filename}")
    
    # Clean up
    shutil.rmtree(dist_dir)
    print(f"Temporary directory {dist_dir} removed.")

if __name__ == "__main__":
    create_distribution_package()
