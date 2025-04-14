"""
Helper script to update monkey_patch imports in test files.
"""

import os
import re

def update_imports():
    """Update monkey_patch imports in all test files."""
    test_files = [f for f in os.listdir('.') if f.endswith('.py') and f != '__init__.py' and f != 'update_monkey_patch.py']
    
    for filename in test_files:
        with open(filename, 'r') as file:
            content = file.read()
        
        # Update the monkey_patch import
        updated_content = re.sub(
            r'import monkey_patch',
            r'from src.utils import monkey_patch',
            content
        )
        
        with open(filename, 'w') as file:
            file.write(updated_content)
        
        print(f"Updated monkey_patch import in {filename}")

if __name__ == "__main__":
    update_imports()
