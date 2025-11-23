#!/usr/bin/env python3
"""
Combined Face and Gesture Recognition System
Integrates the OpenCV LBPH face recognition with gesture recognition.
"""

import argparse
import time
import cv2
import numpy as np
import json
import os
import logging
import sys

# Add the face recognition module to path
sys.path.append('real-time-face-recognition-master/src')

from main_controller import MainController
from utils import Drawer, Event, targets

# Import face recognition settings
try:
    from settings.settings import CAMERA, FACE_DETECTION, PATHS
except ImportError:
    # Fallback settings if import fails
    CAMERA = {'index': 0, 'width': 640, 'height': 480}
    FACE_DETECTION = {'scale_factor': 1.3, 'min_neighbors': 5, 'min_size': (30, 30)}
    PATHS = {
        'cascade_file': 'real-time-face-recognition-master/haarcascade_frontalface_default.xml',
        'names_file': 'real-time-face-recognition-master/names.json',
        'trainer_file': 'real-time-face-recognition-master/trainer.yml'
    }

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedFaceGestureRecognizer:
    """Combined face and gesture recognition system."""
    
    def __init__(self, gesture_detector_path, gesture_classifier_path):
        """
        Initialize the combined recognition system.
        
        Parameters
        ----------
        gesture_detector_path : str
            Path to hand detector ONNX model
        gesture_classifier_path : str
            Path to gesture classifier ONNX model
        """
        # Initialize gesture recognition
        self.gesture_controller = MainController(gesture_detector_path, gesture_classifier_path)
        self.drawer = Drawer()
        
        # Initialize face recognition
        self.face_recognizer = None
        self.face_cascade = None
        self.face_names = {}
        
        self._initialize_face_recognition()
        
        logger.info("Combined Face & Gesture Recognition System initialized!")
    
    def _initialize_face_recognition(self):
        """Initialize the face recognition components."""
        try:
            # Load face recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            trainer_path = PATHS['trainer_file']
            if not os.path.exists(trainer_path):
                # Try alternative path
                trainer_path = f"real-time-face-recognition-master/{PATHS['trainer_file']}"
            
            if os.path.exists(trainer_path):
                self.face_recognizer.read(trainer_path)
                logger.info(f"Face model loaded from: {trainer_path}")
            else:
                logger.warning("Face trainer file not found. Face recognition will be disabled.")
                self.face_recognizer = None
            
            # Load face cascade
            cascade_path = PATHS['cascade_file']
            if not os.path.exists(cascade_path):
                # Try alternative path
                cascade_path = f"real-time-face-recognition-master/{PATHS['cascade_file']}"
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Face cascade not loaded. Face detection will be disabled.")
                self.face_cascade = None
            else:
                logger.info(f"Face cascade loaded from: {cascade_path}")
            
            # Load names
            names_path = PATHS['names_file']
            if not os.path.exists(names_path):
                # Try alternative path
                names_path = f"real-time-face-recognition-master/{PATHS['names_file']}"
            
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.face_names = json.loads(content)
                        logger.info(f"Loaded {len(self.face_names)} known faces")
            else:
                logger.warning("Names file not found. Using default names.")
                
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            self.face_recognizer = None
            self.face_cascade = None
    
    def detect_faces(self, frame):
        """
        Detect and recognize faces in the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
            
        Returns
        -------
        face_results : list
            List of dictionaries containing face information
        """
        face_results = []
        
        if self.face_cascade is None or self.face_recognizer is None:
            return face_results
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                # Recognize the face
                face_roi = gray[y:y+h, x:x+w]
                id, confidence = self.face_recognizer.predict(face_roi)
                
                # Determine name based on confidence
                if confidence <= 100:  # Confidence threshold
                    name = self.face_names.get(str(id), f"ID_{id}")
                else:
                    name = "Unknown"
                
                face_info = {
                    'bbox': (x, y, w, h),
                    'name': name,
                    'confidence': confidence,
                    'id': id
                }
                face_results.append(face_info)
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
        
        return face_results
    
    def process_frame(self, frame):
        """
        Process a single frame for both face and gesture recognition.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
            
        Returns
        -------
        results : dict
            Dictionary containing all detection results
        """
        results = {
            'faces': [],
            'gestures': {
                'bboxes': None,
                'ids': None,
                'labels': None,
                'actions': []
            }
        }
        
        # 1. Face Recognition
        face_results = self.detect_faces(frame)
        results['faces'] = face_results
        
        # 2. Gesture Recognition
        bboxes, ids, labels = self.gesture_controller(frame)
        results['gestures']['bboxes'] = bboxes
        results['gestures']['ids'] = ids
        results['gestures']['labels'] = labels
        
        # 3. Check for gesture actions and associate with faces if possible
        if len(self.gesture_controller.tracks) > 0:
            for trk in self.gesture_controller.tracks:
                if trk["tracker"].time_since_update < 1 and trk["hands"].action is not None:
                    # Try to associate gesture with nearby face
                    associated_face = self._associate_gesture_with_face(trk, face_results)
                    
                    action_info = {
                        'action': trk["hands"].action,
                        'track_id': trk["tracker"].id,
                        'associated_face': associated_face
                    }
                    results['gestures']['actions'].append(action_info)
                    trk["hands"].action = None  # Reset action
        
        return results
    
    def _associate_gesture_with_face(self, gesture_track, face_results):
        """
        Try to associate a gesture with a nearby face.
        
        Parameters
        ----------
        gesture_track : dict
            Gesture track information
        face_results : list
            List of detected faces
            
        Returns
        -------
        associated_face : dict or None
            Face information if association found, None otherwise
        """
        if not face_results or len(gesture_track['hands']) == 0:
            return None
        
        # Get gesture position
        last_hand = gesture_track['hands'][-1]
        if last_hand.bbox is None:
            return None
        
        gesture_center = last_hand.center
        min_distance = float('inf')
        closest_face = None
        
        # Find closest face
        for face in face_results:
            face_x, face_y, face_w, face_h = face['bbox']
            face_center = (face_x + face_w // 2, face_y + face_h // 2)
            
            # Calculate distance between gesture and face
            distance = np.sqrt((gesture_center[0] - face_center[0])**2 + 
                             (gesture_center[1] - face_center[1])**2)
            
            # Only associate if gesture is reasonably close to face (within 200 pixels)
            if distance < 200 and distance < min_distance:
                min_distance = distance
                closest_face = face
        
        return closest_face
    
    def draw_annotations(self, frame, results, debug_mode=True):
        """
        Draw face and gesture annotations on the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        results : dict
            Detection results
        debug_mode : bool
            Whether to show debug information
            
        Returns
        -------
        annotated_frame : np.ndarray
            Frame with annotations
        """
        annotated_frame = frame.copy()
        
        # Draw face annotations
        for face in results['faces']:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            # Draw face rectangle
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw name
            cv2.putText(annotated_frame, name, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw confidence if in debug mode
            if debug_mode:
                confidence_text = f"{confidence:.1f}%"
                cv2.putText(annotated_frame, confidence_text, (x + 5, y + h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Draw gesture annotations
        if debug_mode and results['gestures']['bboxes'] is not None:
            bboxes = results['gestures']['bboxes'].astype(np.int32)
            ids = results['gestures']['ids']
            labels = results['gestures']['labels']
            
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
                # Only show our filtered gestures
                if labels[i] is not None and labels[i] in [27, 32, 35]:
                    gesture = targets[labels[i]]
                    cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                    cv2.putText(
                        annotated_frame,
                        f"Hand {ids[i]} : {gesture}",
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
        
        # Draw gesture actions as text overlay
        y_offset = 60
        for action_info in results['gestures']['actions']:
            action = action_info['action']
            associated_face = action_info.get('associated_face')
            
            if Event.TAP == action:
                gesture_name = "PEACE"
            elif Event.SWIPE_RIGHT == action:
                gesture_name = "THUMBS UP"
            elif Event.SWIPE_DOWN == action:
                gesture_name = "STOP"
            else:
                gesture_name = f"Action: {action}"
            
            # Create message with or without face association
            if associated_face:
                text = f"{associated_face['name']}: {gesture_name}"
                color = (0, 255, 255)  # Yellow for associated gestures
            else:
                text = f"{gesture_name} gesture detected!"
                color = (255, 255, 255)  # White for unassociated gestures
            
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        return annotated_frame


def run_combined_demo(args):
    """Run the combined face and gesture recognition demo."""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize combined system
    system = CombinedFaceGestureRecognizer(
        gesture_detector_path=args.detector,
        gesture_classifier_path=args.classifier
    )
    
    print("Starting Combined Face & Gesture Recognition Demo")
    print("Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 's' to toggle debug mode")
    print("=" * 50)
    
    debug_mode = args.debug
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror the frame
        start_time = time.time()
        
        # Process frame for both face and gesture recognition
        results = system.process_frame(frame)
        
        # Draw annotations
        annotated_frame = system.draw_annotations(frame, results, debug_mode=debug_mode)
        
        # Add FPS counter
        if debug_mode:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Print detection results to console
        for action_info in results['gestures']['actions']:
            action = action_info['action']
            associated_face = action_info.get('associated_face')
            
            if Event.TAP == action:
                gesture_name = "PEACE"
            elif Event.SWIPE_RIGHT == action:
                gesture_name = "THUMBS UP"
            elif Event.SWIPE_DOWN == action:
                gesture_name = "STOP"
            else:
                gesture_name = f"Action: {action}"
            
            if associated_face:
                print(f"[PERSON] {associated_face['name']} made a {gesture_name} gesture!")
            else:
                print(f"[GESTURE] {gesture_name} gesture detected!")
        
        # Show frame
        cv2.imshow("Combined Face & Gesture Recognition", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # Toggle debug mode
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Combined recognition system stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Face and Gesture Recognition Demo")
    
    parser.add_argument(
        "--detector",
        default='models/hand_detector.onnx',
        type=str,
        help="Path to hand detector onnx model"
    )
    
    parser.add_argument(
        "--classifier",
        default='models/crops_classifier.onnx',
        type=str,
        help="Path to gesture classifier onnx model"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with bounding boxes and FPS"
    )
    
    args = parser.parse_args()
    run_combined_demo(args)