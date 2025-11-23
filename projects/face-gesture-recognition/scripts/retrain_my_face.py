#!/usr/bin/env python3
"""
Retrain face recognition for EAC22016 (ID: 1)
This script will capture new images and retrain the model.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import sys
import logging

# Add the face recognition module to path
sys.path.append('real-time-face-recognition-master/src')

try:
    from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS
    # Fix paths to be absolute
    PATHS['cascade_file'] = 'real-time-face-recognition-master/haarcascade_frontalface_default.xml'
    PATHS['names_file'] = 'real-time-face-recognition-master/names.json'
    PATHS['trainer_file'] = 'real-time-face-recognition-master/trainer.yml'
    PATHS['image_dir'] = 'real-time-face-recognition-master/images'
except ImportError:
    # Fallback settings
    CAMERA = {'index': 0, 'width': 640, 'height': 480}
    FACE_DETECTION = {'scale_factor': 1.3, 'min_neighbors': 5, 'min_size': (30, 30)}
    TRAINING = {'samples_needed': 120}
    PATHS = {
        'cascade_file': 'real-time-face-recognition-master/haarcascade_frontalface_default.xml',
        'names_file': 'real-time-face-recognition-master/names.json',
        'trainer_file': 'real-time-face-recognition-master/trainer.yml',
        'image_dir': 'real-time-face-recognition-master/images'
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def delete_old_images(face_id, image_dir):
    """Delete old images for the specified face ID."""
    try:
        if not os.path.exists(image_dir):
            logger.warning(f"Image directory {image_dir} does not exist")
            return 0
            
        deleted = 0
        for filename in os.listdir(image_dir):
            if filename.startswith(f'Users-{face_id}-'):
                filepath = os.path.join(image_dir, filename)
                os.remove(filepath)
                deleted += 1
        
        logger.info(f"Deleted {deleted} old images for ID {face_id}")
        return deleted
    except Exception as e:
        logger.error(f"Error deleting old images: {e}")
        return 0

def capture_face_images(face_id, face_name, samples_needed=120):
    """Capture face images for training."""
    try:
        # Initialize camera
        cam = cv2.VideoCapture(CAMERA['index'])
        if not cam.isOpened():
            raise ValueError("Could not open webcam")
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        
        # Create image directory if needed
        os.makedirs(PATHS['image_dir'], exist_ok=True)
        
        print("\n" + "="*60)
        print(f"FACE CAPTURE FOR: {face_name} (ID: {face_id})")
        print("="*60)
        print(f"\nCapturing {samples_needed} images...")
        print("Instructions:")
        print("  - Look directly at the camera")
        print("  - Move your head slightly (left, right, up, down)")
        print("  - Try different expressions")
        print("  - Keep your face in the blue rectangle")
        print("  - Press ESC to cancel")
        print("\nStarting in 3 seconds...")
        print("="*60 + "\n")
        
        # Wait a moment
        cv2.namedWindow('Face Capture - EAC22016')
        for i in range(3, 0, -1):
            ret, img = cam.read()
            if ret:
                cv2.putText(img, f"Starting in {i}...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Face Capture - EAC22016', img)
                cv2.waitKey(1000)
        
        count = 0
        last_save_time = 0
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            # Draw instructions
            cv2.putText(img, f"EAC22016 - Face Retraining", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for (x, y, w, h) in faces:
                if count < samples_needed:
                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Save face image
                    face_img = gray[y:y+h, x:x+w]
                    img_path = os.path.join(PATHS['image_dir'], f'Users-{face_id}-{count+1}.jpg')
                    cv2.imwrite(img_path, face_img)
                    
                    count += 1
                    
                    # Progress bar
                    progress_pct = int((count / samples_needed) * 100)
                    bar_width = 400
                    bar_filled = int((count / samples_needed) * bar_width)
                    
                    cv2.rectangle(img, (10, 60), (10 + bar_width, 90), (100, 100, 100), -1)
                    cv2.rectangle(img, (10, 60), (10 + bar_filled, 90), (0, 255, 0), -1)
                    
                    progress_text = f"Progress: {count}/{samples_needed} ({progress_pct}%)"
                    cv2.putText(img, progress_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Instructions
                    if count < 30:
                        cv2.putText(img, "Look straight ahead", (10, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    elif count < 60:
                        cv2.putText(img, "Turn head slightly left/right", (10, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    elif count < 90:
                        cv2.putText(img, "Tilt head up/down slightly", (10, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    else:
                        cv2.putText(img, "Try different expressions", (10, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(img, "COMPLETE! Processing...", (10, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    break
            
            if len(faces) == 0:
                cv2.putText(img, "No face detected - please face the camera", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Capture - EAC22016', img)
            
            key = cv2.waitKey(100) & 0xff
            if key == 27:  # ESC
                print("\nCapture cancelled by user")
                cam.release()
                cv2.destroyAllWindows()
                return False
            
            if count >= samples_needed:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Successfully captured {count} images for {face_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error during capture: {e}")
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
        return False

def train_model():
    """Train the face recognition model with all available images."""
    try:
        logger.info("Training face recognition model...")
        
        # Initialize recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load face detector
        detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        if detector.empty():
            raise ValueError("Error loading cascade classifier")
        
        # Get all image paths
        image_paths = [os.path.join(PATHS['image_dir'], f) 
                      for f in os.listdir(PATHS['image_dir']) 
                      if f.startswith('Users-')]
        
        if not image_paths:
            raise ValueError("No training images found")
        
        face_samples = []
        ids = []
        
        for image_path in image_paths:
            # Convert to grayscale
            pil_img = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_img, 'uint8')
            
            # Extract user ID from filename
            filename = os.path.split(image_path)[-1]
            user_id = int(filename.split("-")[1])
            
            # Detect faces
            faces = detector.detectMultiScale(img_numpy)
            
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(user_id)
        
        if not face_samples:
            raise ValueError("No faces detected in training images")
        
        # Train the model
        logger.info(f"Training with {len(face_samples)} face samples from {len(np.unique(ids))} people...")
        recognizer.train(face_samples, np.array(ids))
        
        # Save the model
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model saved to {PATHS['trainer_file']}")
        logger.info(f"Training complete! Trained on IDs: {sorted(np.unique(ids))}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

def main():
    """Main retraining workflow."""
    face_id = 1
    face_name = "EAC22074"
    
    print("\n" + "="*60)
    print("FACE RECOGNITION RETRAINING")
    print("="*60)
    print(f"User: {face_name}")
    print(f"ID: {face_id}")
    print("="*60)
    
    # Step 1: Delete old images
    print("\nStep 1: Removing old training images...")
    deleted = delete_old_images(face_id, PATHS['image_dir'])
    print(f"Removed {deleted} old images")
    
    # Step 2: Capture new images
    print("\nStep 2: Capturing new face images...")
    if not capture_face_images(face_id, face_name, TRAINING['samples_needed']):
        print("\n[FAIL] Capture failed or cancelled")
        return
    
    print("\n[OK] Image capture complete!")
    
    # Step 3: Retrain model
    print("\nStep 3: Retraining face recognition model...")
    if not train_model():
        print("\n[FAIL] Training failed")
        return
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE!")
    print("="*60)
    print(f"\nYour face ({face_name}) has been retrained successfully!")
    print("You can now run the combined system:")
    print("  python combined_face_gesture_recognition.py --debug")
    print("  python capture_report_images.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
