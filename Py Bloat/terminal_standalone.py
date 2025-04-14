"""
Standalone Terminal UI Entry Point for 3lacks Scanner

This file applies the numpy monkey patch and then launches the terminal UI.
It's designed to work when double-clicked outside of a terminal.
"""

# Apply patches first
import sys
import os
import traceback

# Make sure we're in the PyBloat directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add the current directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Try to patch typing_extensions
try:
    from scripts.build.hooks import typing_extensions_patch
    typing_extensions_patch.patch_typing_extensions()
except ImportError:
    print("Warning: typing_extensions_patch not found")

# Import the numpy monkey patch
try:
    from src.utils import monkey_patch
except ImportError:
    print("Warning: monkey_patch not found")
    # Try to import from the current directory
    try:
        import monkey_patch
    except ImportError:
        print("Error: monkey_patch not found in any location")

# Set up error handling
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions."""
    # Log the error
    print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Keep the console window open
    input("\nPress Enter to exit...")
    return sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set the exception hook
sys.excepthook = handle_exception

def main():
    """Main function to launch the terminal UI."""
    try:
        # Import the terminal UI
        from src.ui.terminal_ui import main as terminal_main
        terminal_main()
    except ImportError as e:
        print(f"Error importing terminal UI: {e}")
        print("Trying alternative import paths...")
        
        # Try to import from the parent directory
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from src.ui.terminal_ui import main as terminal_main
            terminal_main()
        except ImportError as e:
            print(f"Error importing terminal UI from parent directory: {e}")
            print("Could not find the terminal UI module.")
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Keep the console window open if there's an error
    try:
        main()
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
