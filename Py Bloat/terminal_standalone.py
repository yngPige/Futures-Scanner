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

# Get the parent directory (project root)
parent_dir = os.path.dirname(script_dir)

# Change to the parent directory
os.chdir(parent_dir)

# Add both directories to the Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
    # Apply the patch
    monkey_patch.apply_patches()
except ImportError:
    print("Warning: monkey_patch not found")
    # Try to import from the current directory
    try:
        import monkey_patch
        # Apply the patch
        monkey_patch.apply_patches()
    except (ImportError, AttributeError):
        print("Error: monkey_patch not found or could not be applied")

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
        # Print the current directory and sys.path for debugging
        print(f"Current directory: {os.getcwd()}")
        print(f"Parent directory: {parent_dir}")
        print(f"Script directory: {script_dir}")
        print(f"sys.path: {sys.path}")

        # Check if terminal_ui.py exists
        terminal_ui_path = os.path.join(parent_dir, 'src', 'ui', 'terminal_ui.py')
        if os.path.exists(terminal_ui_path):
            print(f"Found terminal_ui.py at {terminal_ui_path}")
        else:
            print(f"Could not find terminal_ui.py at {terminal_ui_path}")
            # List files in the directory
            src_ui_dir = os.path.join(parent_dir, 'src', 'ui')
            if os.path.exists(src_ui_dir):
                print(f"Files in {src_ui_dir}:")
                for file in os.listdir(src_ui_dir):
                    print(f"  {file}")
            else:
                print(f"Directory {src_ui_dir} does not exist")

        # Try to import the terminal UI
        try:
            # Import the terminal UI
            from src.ui.terminal_ui import main as terminal_main
            terminal_main()
        except ImportError as e:
            print(f"Error importing terminal UI: {e}")
            print("Trying alternative import paths...")

            # Try to import the module directly using importlib
            try:
                # Import the module directly using the full path
                import importlib.util

                if os.path.exists(terminal_ui_path):
                    print("Importing terminal_ui.py directly using importlib...")
                    spec = importlib.util.spec_from_file_location("terminal_ui", terminal_ui_path)
                    terminal_ui = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(terminal_ui)

                    # Call the main function
                    print("Calling terminal_ui.main()...")
                    terminal_ui.main()
                else:
                    print(f"Could not find terminal_ui.py at {terminal_ui_path}")
                    print("Could not find the terminal UI module.")
            except Exception as e:
                print(f"Error importing terminal UI directly: {e}")
                traceback.print_exc()
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
