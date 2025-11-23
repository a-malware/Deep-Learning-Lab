#!/usr/bin/env python3
"""
Verify that the system is properly set up and all dependencies are available.
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"[FAIL] Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'onnxruntime': 'onnxruntime',
        'scipy': 'scipy',
        'PIL': 'Pillow'
    }
    
    all_installed = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} (not installed)")
            all_installed = False
    
    return all_installed

def check_files():
    """Check if required files exist."""
    required_files = [
        'combined_face_gesture_recognition.py',
        'main_controller.py',
        'onnx_models.py',
        'models/hand_detector.onnx',
        'models/crops_classifier.onnx',
        'real-time-face-recognition-master/haarcascade_frontalface_default.xml',
        'real-time-face-recognition-master/names.json'
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"[OK] {filepath}")
        else:
            print(f"[FAIL] {filepath} (missing)")
            all_exist = False
    
    return all_exist

def check_camera():
    """Check if camera is accessible."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[OK] Camera accessible")
            cap.release()
            return True
        else:
            print("[FAIL] Camera not accessible")
            return False
    except Exception as e:
        print(f"[FAIL] Camera check failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("SYSTEM VERIFICATION")
    print("="*60 + "\n")
    
    print("Checking Python version...")
    python_ok = check_python_version()
    print()
    
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("Checking required files...")
    files_ok = check_files()
    print()
    
    print("Checking camera...")
    camera_ok = check_camera()
    print()
    
    print("="*60)
    if python_ok and deps_ok and files_ok and camera_ok:
        print("ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run the system:")
        print("  python combined_face_gesture_recognition.py --debug")
    else:
        print("SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the system.")
        if not deps_ok:
            print("\nInstall missing dependencies:")
            print("  pip install -r requirements.txt")
    print()

if __name__ == "__main__":
    main()
