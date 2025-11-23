#!/usr/bin/env python3
"""
Simple launcher script for the Combined Face & Gesture Recognition System.
"""

import sys
import subprocess

def main():
    """Launch the combined recognition system with appropriate arguments."""
    
    print("Combined Face & Gesture Recognition System")
    print("=" * 50)
    
    # Default arguments
    args = [
        "python", 
        "combined_face_gesture_recognition.py",
        "--debug"  # Enable debug mode by default
    ]
    
    # Check if user wants to disable debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "--no-debug":
        args.remove("--debug")
        print("Starting in normal mode (no debug visualization)")
    else:
        print("Starting in debug mode (with bounding boxes and FPS)")
    
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 's' to toggle debug mode")
    print("=" * 50)
    
    try:
        # Launch the main application
        subprocess.run(args)
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()