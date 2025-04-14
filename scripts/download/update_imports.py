"""
Helper script to update imports in download scripts.
"""

import os
import re

def update_imports():
    """Update imports in all download scripts."""
    script_files = [f for f in os.listdir('.') if f.endswith('.py') and f != '__init__.py' and f != 'update_imports.py']
    
    for filename in script_files:
        with open(filename, 'r') as file:
            content = file.read()
        
        # Update the import paths
        updated_content = re.sub(
            r'from src\.analysis\.local_llm',
            r'from src.analysis.local_llm',
            content
        )
        
        # Add sys.path.append if not already present
        if 'sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))' not in updated_content:
            updated_content = re.sub(
                r'import sys\nimport os',
                r'import sys\nimport os\n\n# Add the project root directory to sys.path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))',
                updated_content
            )
        
        with open(filename, 'w') as file:
            file.write(updated_content)
        
        print(f"Updated imports in {filename}")

if __name__ == "__main__":
    update_imports()
