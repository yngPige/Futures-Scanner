#!/usr/bin/env python
"""
Wrapper script to run any of the organized scripts in the 3lacks Scanner project.
"""

import os
import sys
import subprocess

def main():
    """Main function to forward arguments to the actual run_script.py."""
    # Get the path to the scripts directory
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')

    # Change to the scripts directory before running the script
    os.chdir(scripts_dir)

    # Forward all arguments to the actual run_script.py
    cmd = [sys.executable, 'run_script.py'] + sys.argv[1:]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
