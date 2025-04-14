#!/usr/bin/env python
"""
Script to clean up old files after the reorganization.

This script removes the original files that have been moved to new locations
during the file structure reorganization.
"""

import os
import sys
import shutil

# Files to remove
FILES_TO_REMOVE = [
    # Original script files
    "build_executable.py",
    "build_executable.bat",
    "create_distribution.py",
    "create_distribution.bat",
    "create_fixed_distribution.py",
    "create_fixed_distribution.bat",
    "create_icon.py",
    "download_llama3_model.py",
    "download_llm_model.py",
    "download_model.py",
    "download_model.bat",
    "download_phi3_mini.py",
    "fix_build.py",
    "fix_build.bat",
    "fix_model_download.py",
    "fix_model_download_curl.py",
    "fix_model_download_hf.py",
    "fix_model_download_phi3.py",
    
    # Original documentation files
    "FUNCTION_KEYS.md",
    "README_EXECUTABLE.md",
    "README_LLM.md",
    "TERMINAL_UI_GUIDE.md",
    
    # Original setup files
    "install_dependencies.bat",
    "install_keyboard.bat",
    "install_llm_deps.py",
    "install_pyinstaller.py",
    
    # Original asset files
    "icon.ico",
    "null.png",
    
    # Original build files
    "hook-dependencies.py",
    "hook-typing_extensions.py",
    "rth_typing_extensions.py",
    "typing_extensions_patch.py",
    "simple_build.py",
    "simple_build.bat",
    
    # Original monkey patch
    "monkey_patch.py",
    
    # Reorganization notice
    "REORGANIZATION_NOTICE.md"
]

def main():
    """Remove the original files that have been moved to new locations."""
    # Get confirmation from the user
    print("This script will remove the original files that have been moved to new locations.")
    print("Make sure you have updated all references to these files before proceeding.")
    print()
    print("Files to remove:")
    for file in FILES_TO_REMOVE:
        if os.path.exists(os.path.join("..", file)):
            print(f"  - {file}")
    print()
    
    confirmation = input("Are you sure you want to remove these files? (y/n): ")
    if confirmation.lower() != "y":
        print("Operation cancelled.")
        return
    
    # Remove the files
    for file in FILES_TO_REMOVE:
        file_path = os.path.join("..", file)
        if os.path.exists(file_path):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    print()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
