#!/usr/bin/env python
"""
Utility runner for 3lacks Scanner.

This script provides a command-line interface to run various utility functions.
"""

import os
import sys
import argparse
import importlib.util

def list_available_modules():
    """List all available utility modules."""
    modules = {}
    
    # Define the utility categories
    categories = ['data', 'models', 'build', 'setup', 'visualization']
    
    for category in categories:
        modules[category] = []
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), category)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.py') and file != '__init__.py':
                    modules[category].append(file)
    
    return modules

def run_module(module_path):
    """Run a module from the given path."""
    # Get the full path to the module
    full_path = os.path.abspath(module_path)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check if the module has a main function
    if hasattr(module, 'main'):
        module.main()
    # Check if the module has a run function
    elif hasattr(module, 'run'):
        module.run()
    # Otherwise, the module has already been executed when imported
    else:
        print(f"Module {module_path} does not have a main or run function.")

def main():
    """Main function to parse arguments and run utility modules."""
    parser = argparse.ArgumentParser(description='3lacks Scanner Utility Runner')
    parser.add_argument('--list', action='store_true', help='List all available utility modules')
    parser.add_argument('--category', type=str, help='Utility category (data, models, build, setup, visualization)')
    parser.add_argument('--module', type=str, help='Module name to run')
    
    args = parser.parse_args()
    
    # List all available modules
    if args.list:
        modules = list_available_modules()
        print("Available utility modules:")
        for category, module_list in modules.items():
            print(f"\n{category.upper()}:")
            for module in module_list:
                print(f"  - {module}")
        return
    
    # Run a specific module
    if args.category and args.module:
        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.category, args.module)
        if os.path.exists(module_path):
            print(f"Running {module_path}...")
            run_module(module_path)
        else:
            print(f"Error: Module {module_path} not found.")
            return
    elif args.category:
        print(f"Error: Please specify a module to run with --module")
    elif args.module:
        print(f"Error: Please specify a category with --category")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
