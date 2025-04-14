#!/usr/bin/env python
"""
Create a Windows shortcut for 3lacks Scanner.

This script creates a Windows shortcut (.lnk) file that launches the 3lacks Scanner
in a new console window.
"""

import os
import sys
import winshell
from win32com.client import Dispatch

def create_shortcut():
    """Create a Windows shortcut for 3lacks Scanner."""
    # Get the path to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the path to the PyBloat directory
    pybloat_dir = os.path.join(current_dir, 'Py Bloat')
    
    # Get the path to the terminal.py script
    terminal_path = os.path.join(pybloat_dir, 'terminal.py')
    
    # Get the path to the Python executable
    python_exe = sys.executable
    
    # Create the shortcut
    desktop = winshell.desktop()
    path = os.path.join(desktop, "3lacks Scanner.lnk")
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = "cmd.exe"
    shortcut.Arguments = f"/k {python_exe} {terminal_path}"
    shortcut.WorkingDirectory = pybloat_dir
    shortcut.IconLocation = os.path.join(pybloat_dir, "icon.ico")
    shortcut.save()
    
    print(f"Shortcut created at: {path}")
    
    # Create a shortcut in the current directory as well
    path = os.path.join(current_dir, "3lacks Scanner.lnk")
    shortcut = shell.CreateShortCut(path)
    shortcut.Targetpath = "cmd.exe"
    shortcut.Arguments = f"/k {python_exe} {terminal_path}"
    shortcut.WorkingDirectory = pybloat_dir
    shortcut.IconLocation = os.path.join(pybloat_dir, "icon.ico")
    shortcut.save()
    
    print(f"Shortcut created at: {path}")

if __name__ == "__main__":
    try:
        create_shortcut()
    except Exception as e:
        print(f"Error creating shortcut: {str(e)}")
        input("Press Enter to exit...")
