#!/usr/bin/env python
"""
Notice about the reorganization of utility scripts in the 3lacks Scanner project.
"""

import os
import sys
import time

def main():
    """Display a notice about the reorganization of utility scripts."""
    print("=" * 80)
    print("NOTICE: Utility Scripts Reorganization".center(80))
    print("=" * 80)
    print()
    print("The utility scripts have been reorganized into the 'scripts' directory:")
    print()
    print("  - Download scripts: scripts/download/")
    print("  - Fix model scripts: scripts/fix_model/")
    print("  - Create scripts: scripts/create/")
    print("  - Build scripts: scripts/build/")
    print()
    print("To run these scripts, use the new run_script.py utility:")
    print()
    print("  python run_script.py --list")
    print("  python run_script.py --category download --script download_llm_model.py")
    print()
    print("The original script files in the root directory will be removed in a future update.")
    print("Please update any references to these scripts.")
    print()
    print("=" * 80)
    
    # Wait for user acknowledgment
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()
