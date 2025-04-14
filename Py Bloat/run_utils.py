#!/usr/bin/env python
"""
Wrapper script to run utility functions for 3lacks Scanner.
"""

import os
import sys
import subprocess

def main():
    """Main function to forward arguments to the actual run_utils.py."""
    # Get the path to the utils directory
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
    run_utils_path = os.path.join(utils_dir, 'run_utils.py')
    
    # Forward all arguments to the actual run_utils.py
    cmd = [sys.executable, run_utils_path] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
