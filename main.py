#!/usr/bin/env python3
"""
Main entry point for Ustad standalone executable.

This script handles the PyInstaller environment and starts the server.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
if getattr(sys, 'frozen', False):
    # PyInstaller executable
    bundle_dir = Path(sys._MEIPASS)
    sys.path.insert(0, str(bundle_dir))
else:
    # Normal Python execution
    bundle_dir = Path(__file__).parent

# Change to the bundle directory
os.chdir(bundle_dir)

# Now import and run the server
if __name__ == "__main__":
    try:
        from backend.server import main
        main()
    except Exception as e:
        print(f"Error starting Ustad: {e}")
        input("Press Enter to exit...")
        sys.exit(1)