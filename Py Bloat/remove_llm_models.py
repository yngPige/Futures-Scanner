"""
Script to remove all LLM models from the cache directory.
"""

import os
import glob
import shutil
import sys
from colorama import init, Fore, Style

# Initialize colorama
init()

# Default model path
DEFAULT_MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "futures_scanner", "models")

def print_info(message):
    """Print an info message."""
    print(Fore.BLUE + message + Style.RESET_ALL)

def print_success(message):
    """Print a success message."""
    print(Fore.GREEN + message + Style.RESET_ALL)

def print_error(message):
    """Print an error message."""
    print(Fore.RED + message + Style.RESET_ALL)

def print_warning(message):
    """Print a warning message."""
    print(Fore.YELLOW + message + Style.RESET_ALL)

def remove_all_models():
    """Remove all LLM models from the cache directory."""
    print("\n" + "=" * 80)
    print(Fore.CYAN + "Remove All LLM Models".center(80) + Style.RESET_ALL)
    print("=" * 80 + "\n")
    
    # Check if the models directory exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print_info(f"No models directory found at {DEFAULT_MODEL_PATH}")
        return
    
    # Get a list of all model files
    model_files = glob.glob(os.path.join(DEFAULT_MODEL_PATH, "*.gguf"))
    model_files += glob.glob(os.path.join(DEFAULT_MODEL_PATH, "*.ggml"))
    model_files += glob.glob(os.path.join(DEFAULT_MODEL_PATH, "*.bin"))
    
    if not model_files:
        print_info("No model files found.")
        return
    
    # Display the list of models to be removed
    print_info(f"Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files, 1):
        file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        file_name = os.path.basename(model_file)
        print_info(f"{i}. {file_name} ({file_size_mb:.2f} MB)")
    
    # Ask for confirmation
    confirm = input(Fore.GREEN + "\nAre you sure you want to remove ALL LLM models? This cannot be undone. (y/n): " + Style.RESET_ALL)
    if confirm.lower() != 'y':
        print_info("Operation cancelled.")
        return
    
    # Show a second confirmation for safety
    confirm2 = input(Fore.GREEN + "\nFinal confirmation - remove ALL models? (type 'CONFIRM' to proceed): " + Style.RESET_ALL)
    if confirm2 != 'CONFIRM':
        print_info("Operation cancelled.")
        return
    
    # Remove all model files
    removed_count = 0
    failed_count = 0
    
    print_info("\nRemoving model files...")
    
    for model_file in model_files:
        try:
            os.remove(model_file)
            print_success(f"Removed: {os.path.basename(model_file)}")
            removed_count += 1
        except Exception as e:
            print_error(f"Failed to remove {os.path.basename(model_file)}: {e}")
            failed_count += 1
    
    # Display results
    if removed_count > 0:
        print_success(f"\nSuccessfully removed {removed_count} model files.")
    if failed_count > 0:
        print_warning(f"Failed to remove {failed_count} model files.")
    
    # Option to remove the entire directory
    if removed_count > 0 and failed_count == 0:
        remove_dir = input(Fore.GREEN + "\nDo you want to remove the entire models directory? (y/n): " + Style.RESET_ALL)
        if remove_dir.lower() == 'y':
            try:
                shutil.rmtree(DEFAULT_MODEL_PATH)
                print_success(f"Successfully removed directory: {DEFAULT_MODEL_PATH}")
            except Exception as e:
                print_error(f"Failed to remove directory: {e}")

def main():
    """Main function."""
    try:
        remove_all_models()
        print("\nPress Enter to exit...")
        input()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print_error(f"Error: {e}")
        print("\nPress Enter to exit...")
        input()

if __name__ == "__main__":
    main()
