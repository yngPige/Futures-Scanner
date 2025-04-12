"""
Entry point for the Crypto Futures Scanner application.
This file imports the monkey patch for numpy before importing the main module.
"""

import sys
import os

# Print Python version and path
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Import the monkey patch first
import monkey_patch

try:
    # Then import and run the main module
    print("Importing main module...")
    import main
    print("Main module imported successfully.")

    # Check if main has the expected attributes
    print(f"main.__file__: {main.__file__}")
    print(f"main has 'main' function: {'main' in dir(main)}")

    # If main has a main function, call it with sys.argv
    if 'main' in dir(main) and callable(main.main):
        print("Calling main.main()...")
        main.main()
    else:
        print("Warning: main.main() not found or not callable.")
        print(f"dir(main): {dir(main)}")

except Exception as e:
    print(f"Error importing or running main: {e}")
    import traceback
    traceback.print_exc()
