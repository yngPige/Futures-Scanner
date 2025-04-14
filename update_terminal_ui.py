"""
Script to update terminal_ui.py to use PyBloat directory.
"""

import re

def update_terminal_ui():
    """Update terminal_ui.py to use PyBloat directory."""
    # Read the file
    with open('src/ui/terminal_ui.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all direct imports from main with the helper method
    pattern = r'try:\s+from main import'
    replacement = 'try:\n            # Import main from PyBloat directory\n            self._import_from_pybloat()\n            from main import'
    updated_content = re.sub(pattern, replacement, content)
    
    # Write the updated content back to the file
    with open('src/ui/terminal_ui.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Updated terminal_ui.py to use PyBloat directory.")

if __name__ == "__main__":
    update_terminal_ui()
