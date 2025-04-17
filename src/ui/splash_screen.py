"""
Splash Screen Module for Blacks Scanner

This module provides an animated splash screen with ASCII art animations
including a rotating globe, "Blacks Scanner" text, and a loading bar.
"""

import PySimpleGUI as sg
import time
import threading
import random
import sys
import os

# ASCII Art for "Blacks Scanner"
BLACKS_SCANNER_ASCII = """
 /$$$$$$$  /$$                     /$$                    /$$$$$$                                                      
| $$__  $$| $$                    | $$                   /$$__  $$                                                     
| $$  \\ $$| $$  /$$$$$$   /$$$$$$$| $$   /$$  /$$$$$$$ | $$  \\__/  /$$$$$$$  /$$$$$$  /$$$$$$$  /$$$$$$$   /$$$$$$   /$$$$$$
| $$$$$$$ | $$ |____  $$ /$$_____/| $$  /$$/ /$$_____/ |  $$$$$$  /$$_____/ |____  $$| $$__  $$| $$__  $$ /$$__  $$ /$$__  $$
| $$__  $$| $$  /$$$$$$$| $$      | $$$$$$/ |  $$$$$$   \\____  $$| $$        /$$$$$$$| $$  \\ $$| $$  \\ $$| $$$$$$$$| $$  \\__/
| $$  \\ $$| $$ /$$__  $$| $$      | $$_  $$  \\____  $$ /$$  \\ $$| $$       /$$__  $$| $$  | $$| $$  | $$| $$_____/| $$
| $$$$$$$/| $$|  $$$$$$$|  $$$$$$$| $$ \\  $$ /$$$$$$$//  $$$$$$/|  $$$$$$$|  $$$$$$$| $$  | $$| $$  | $$|  $$$$$$$| $$
|_______/ |__/ \\_______/ \\_______/|__/  \\__/|_______/  \\______/  \\_______/ \\_______/|__/  |__/|__/  |__/ \\_______/|__/
"""

# ASCII Art for rotating globe frames
GLOBE_FRAMES = [
    """
    ,---.
   /     \\
  |       |
   \\     /
    `---'
    """,
    """
    ,---.
   /  /  \\
  |  /    |
   \\/     /
    `---'
    """,
    """
    ,---.
   /  |  \\
  |   |   |
   \\  |  /
    `---'
    """,
    """
    ,---.
   /  \\  \\
  |    \\  |
   \\     \\/
    `---'
    """,
]

class SplashScreen:
    """Class to display an animated splash screen."""

    def __init__(self, duration=5):
        """
        Initialize the splash screen.

        Args:
            duration (int): Duration in seconds to display the splash screen
        """
        self.duration = duration
        self.window = None
        self.stop_animation = False
        self.animation_thread = None

    def _create_layout(self):
        """Create the layout for the splash screen."""
        # Set theme to dark
        sg.theme('DarkBlack')

        # Create layout
        layout = [
            [sg.Text(BLACKS_SCANNER_ASCII, font='Courier 8', text_color='#00FF00', background_color='black', pad=(0, 0))],
            [sg.Text('', size=(60, 6), key='-GLOBE-', font='Courier 12', text_color='#00FFFF', background_color='black', pad=(0, 0))],
            [sg.Text('Loading...', key='-LOADING_TEXT-', font='Courier 10', text_color='white', background_color='black')],
            [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-', bar_color=('#00FF00', '#333333'))],
            [sg.Text('', key='-STATUS-', size=(60, 1), font='Courier 10', text_color='#AAAAAA', background_color='black')],
            [sg.Button('Start', key='-START-', button_color=('#000000', '#00FF00'), font='Courier 12', visible=False)]
        ]

        return layout

    def _animate_splash(self):
        """Run the animation for the splash screen."""
        progress = 0
        frame_idx = 0
        status_messages = [
            "Initializing system...",
            "Loading market data...",
            "Preparing technical indicators...",
            "Calibrating prediction models...",
            "Optimizing trading algorithms...",
            "Scanning global markets...",
            "Analyzing historical patterns...",
            "Ready to launch..."
        ]
        
        start_time = time.time()
        
        while not self.stop_animation:
            # Calculate progress based on elapsed time
            elapsed = time.time() - start_time
            progress = min(100, int((elapsed / self.duration) * 100))
            
            # Update globe animation
            if self.window:
                self.window['-GLOBE-'].update(GLOBE_FRAMES[frame_idx])
                frame_idx = (frame_idx + 1) % len(GLOBE_FRAMES)
                
                # Update progress bar
                self.window['-PROGRESS-'].update(progress)
                
                # Update status message
                status_idx = min(len(status_messages) - 1, int(progress / (100 / len(status_messages))))
                self.window['-STATUS-'].update(status_messages[status_idx])
                
                # Show start button when loading is complete
                if progress >= 100:
                    self.window['-LOADING_TEXT-'].update('Ready to launch')
                    self.window['-START-'].update(visible=True)
                    self.stop_animation = True
            
            # Sleep to control animation speed
            time.sleep(0.1)

    def run(self):
        """
        Display the splash screen.

        Returns:
            bool: True if the user clicked Start, False if they closed the window
        """
        # Create the window
        layout = self._create_layout()
        self.window = sg.Window('Blacks Scanner', layout, no_titlebar=True, finalize=True, 
                               background_color='black', keep_on_top=True, 
                               element_justification='center', margins=(0, 0))
        
        # Center the window on screen
        self.window.move(
            (self.window.get_screen_size()[0] - self.window.size[0]) // 2,
            (self.window.get_screen_size()[1] - self.window.size[1]) // 2
        )
        
        # Start animation thread
        self.animation_thread = threading.Thread(target=self._animate_splash, daemon=True)
        self.animation_thread.start()
        
        # Event loop
        result = False
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WINDOW_CLOSED:
                break
            elif event == '-START-':
                result = True
                break
        
        # Clean up
        self.stop_animation = True
        if self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1)
        self.window.close()
        
        return result


# Example usage
if __name__ == "__main__":
    splash = SplashScreen(duration=5)
    result = splash.run()
    
    if result:
        print("Starting application...")
    else:
        print("Splash screen closed without starting.")
