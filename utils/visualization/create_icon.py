"""
Script to create an icon for the 3lacks Scanner application.
"""
import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import os
from PIL import Image, ImageDraw, ImageFont

def create_icon():
    """Create an icon for the application."""
    # Create a new image with a black background
    img = Image.new('RGBA', (256, 256), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a green circle
    draw.ellipse((20, 20, 236, 236), fill=(0, 200, 0, 255))
    
    # Draw a smaller black circle in the center
    draw.ellipse((60, 60, 196, 196), fill=(0, 0, 0, 255))
    
    # Draw "3L" text in the center
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    draw.text((90, 70), "3L", fill=(0, 255, 0, 255), font=font)
    
    # Save as ICO file
    img.save("icon.ico", format="ICO")
    print(f"Icon saved to {os.path.abspath('icon.ico')}")

if __name__ == "__main__":
    create_icon()
