#!/usr/bin/env python
"""
GUI Launcher script for 3lacks Scanner.

This script launches the 3lacks Scanner application using the PyBloat directory.
It's designed to work when double-clicked outside of a terminal.
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

def launch_scanner():
    """Launch the 3lacks Scanner application."""
    # Get the path to the PyBloat directory
    pybloat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Py Bloat')
    
    # Check if the PyBloat directory exists
    if not os.path.exists(pybloat_dir):
        messagebox.showerror("Error", f"PyBloat directory not found at {pybloat_dir}")
        return False
    
    # Change to the PyBloat directory
    os.chdir(pybloat_dir)
    
    # Launch the terminal.py script
    terminal_path = os.path.join(pybloat_dir, 'terminal.py')
    if os.path.exists(terminal_path):
        try:
            # Create a new console window and run the terminal.py script
            if sys.platform == 'win32':
                # On Windows, use the START command to open a new console window
                subprocess.Popen(['start', 'cmd', '/k', sys.executable, terminal_path], shell=True)
            else:
                # On Unix-like systems, try to open a new terminal window
                terminals = ['gnome-terminal', 'xterm', 'konsole', 'terminator']
                terminal_found = False
                
                for terminal in terminals:
                    try:
                        if terminal == 'gnome-terminal':
                            subprocess.Popen([terminal, '--', sys.executable, terminal_path])
                        else:
                            subprocess.Popen([terminal, '-e', f'{sys.executable} {terminal_path}'])
                        terminal_found = True
                        break
                    except FileNotFoundError:
                        continue
                
                if not terminal_found:
                    messagebox.showerror("Error", "Could not find a suitable terminal emulator.")
                    return False
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error launching scanner: {str(e)}")
            return False
    else:
        messagebox.showerror("Error", f"terminal.py not found in {pybloat_dir}")
        return False

def main():
    """Main function to show a splash screen and launch the scanner."""
    # Create a simple splash screen
    root = tk.Tk()
    root.title("3lacks Scanner")
    root.geometry("400x200")
    root.configure(bg="#1e1e1e")
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Add a label with the application name
    title_label = tk.Label(root, text="3lacks Scanner", font=("Arial", 24, "bold"), fg="#ffffff", bg="#1e1e1e")
    title_label.pack(pady=20)
    
    # Add a label with a loading message
    loading_label = tk.Label(root, text="Launching...", font=("Arial", 12), fg="#cccccc", bg="#1e1e1e")
    loading_label.pack(pady=10)
    
    # Add a button to launch the scanner
    launch_button = tk.Button(root, text="Launch Scanner", command=lambda: launch_and_close(root), 
                             font=("Arial", 12), bg="#007acc", fg="white", 
                             activebackground="#005999", activeforeground="white",
                             padx=20, pady=10)
    launch_button.pack(pady=20)
    
    # Function to launch the scanner and close the splash screen
    def launch_and_close(root):
        if launch_scanner():
            root.destroy()
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
