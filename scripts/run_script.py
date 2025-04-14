#!/usr/bin/env python
"""
Main script to run any of the organized scripts in the 3lacks Scanner project.
"""

import os
import sys
import subprocess
import argparse

def list_available_scripts():
    """List all available scripts in the scripts directory."""
    scripts = {}

    # Define the script categories we're interested in
    script_categories = ['download', 'fix_model', 'create', 'build']

    for category in script_categories:
        scripts[category] = []
        dir_path = os.path.join('.', category)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.py') and file != '__init__.py' and file != 'update_imports.py':
                    scripts[category].append(file)

    return scripts

def run_script(script_path, script_args=None):
    """Run a script from the given path."""
    if script_args is None:
        script_args = []

    # Get the full path to the script
    full_path = os.path.abspath(script_path)

    # Run the script as a subprocess to properly handle command-line arguments
    cmd = [sys.executable, full_path] + script_args
    subprocess.run(cmd)

def main():
    """Main function to parse arguments and run scripts."""
    parser = argparse.ArgumentParser(description='Run 3lacks Scanner scripts')
    parser.add_argument('--list', action='store_true', help='List all available scripts')
    parser.add_argument('--category', type=str, help='Script category (download, fix_model, create, build)')
    parser.add_argument('--script', type=str, help='Script name to run')

    # Parse known args to extract our own arguments
    args, remaining_args = parser.parse_known_args()

    # List all available scripts
    if args.list:
        scripts = list_available_scripts()
        print("Available scripts:")
        for category, script_list in scripts.items():
            print(f"\n{category.upper()}:")
            for script in script_list:
                print(f"  - {script}")
        return

    # Run a specific script
    if args.category and args.script:
        script_path = os.path.join('.', args.category, args.script)
        if os.path.exists(script_path):
            print(f"Running {script_path}...")
            run_script(script_path, remaining_args)
        else:
            print(f"Error: Script {script_path} not found.")
            return
    elif args.category:
        print(f"Error: Please specify a script to run with --script")
    elif args.script:
        print(f"Error: Please specify a category with --category")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
