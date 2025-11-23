#!/usr/bin/env python3
"""
Modified version of combined system to capture images for the report.
Press number keys to capture specific scenarios:
1 - Single person with gesture
2 - Multi-person scenario
3 - Gesture examples (peace, thumbs up, stop)
4 - Detection example with bounding boxes
"""

import argparse
import time
import cv2
import numpy as np
import json
import os
import logging
import sys
from datetime import datetime

# Add the face recognition module to path
sys.path.append('real-time-face-recognition-master/src')

from main_controller import MainController
from utils import Drawer, Event, targets

# Import face recognition settings
try:
    from settings.settings import CAMERA, FACE_DETECTION, PATHS
except ImportError:
    CAMERA = {'index': 0, 'width': 640, 'height': 480}
    FACE_DETECTION = {'scale_factor': 1.3, 'min_neighbors': 5, 'min_size': (30, 30)}
    PATHS = {
        'cascade_file': 'real-time-face-recognition-master/haarcascade_frontalface_default.xml',
        'names_file': 'real-time-face-recognition-master/names.json',
        'trainer_file': 'real-time-face-recognition-master/trainer.yml'
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedFaceGestureRecognizer:
    """Combined face and gesture recognition system."""
    
    def __init__(self, gesture_detector_path, gesture_classifier_path):
        self.gesture_controller = MainController(gesture_detector_path, gesture_classifier_path)
        self.drawer = Drawer()
        self.face_recognizer = None
        self.face_cascade = None
        self.face_names = {}
        self._initialize_face_recognition()
        logger.info("Combined Face & Gesture Recognition System initialized!")
    
    def _initialize_face_recognition(self):
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            trainer_path = PATHS['trainer_file']
            if not os.path.exists(trainer_path):
                trainer_path = f"real-time-face-recognition-master/{PATHS['trainer_file']}"
            
            if os.path.exists(trainer_path):
                self.face_recognizer.read(trainer_path)
                logger.info(f"Face model loaded from: {trainer_path}")
            else:
                logger.warning("Face trainer file not found.")
                self.face_recognizer = None
            
            cascade_path = PATHS['cascade_file']
            if not os.path.exists(cascade_path):
                cascade_path = f"real-time-face-recognition-master/{PATHS['cascade_file']}"
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Face cascade not loaded.")
                self.face_cascade = None
            
            names_path = PATHS['names_file']
            if not os.path.exists(names_path):
                names_path = f"real-time-face-recognition-master/{PATHS['names_file']}"
            
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.face_names = json.loads(content)
                        logger.info(f"Loaded {len(self.face_names)} known faces")
                        
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            self.face_recognizer = None
            self.face_cascade = None
    
    def detect_faces(self, frame):
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
                face_roi = gray[y:y+h, x:x+w]
                id, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence <= 100:
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
        results = {
            'faces': [],
            'gestures': {
                'bboxes': None,
                'ids': None,
                'labels': None,
                'actions': []
            }
        }
        
        face_results = self.detect_faces(frame)
        results['faces'] = face_results
        
        bboxes, ids, labels = self.gesture_controller(frame)
        results['gestures']['bboxes'] = bboxes
        results['gestures']['ids'] = ids
        results['gestures']['labels'] = labels
        
        if len(self.gesture_controller.tracks) > 0:
            for trk in self.gesture_controller.tracks:
                if trk["tracker"].time_since_update < 1 and trk["hands"].action is not None:
                    associated_face = self._associate_gesture_with_face(trk, face_results)
                    action_info = {
                        'action': trk["hands"].action,
                        'track_id': trk["tracker"].id,
                        'associated_face': associated_face
                    }
                    results['gestures']['actions'].append(action_info)
                    trk["hands"].action = None
        
        return results
    
    def _associate_gesture_with_face(self, gesture_track, face_results):
        if not face_results or len(gesture_track['hands']) == 0:
            return None
        
        last_hand = gesture_track['hands'][-1]
        if last_hand.bbox is None:
            return None
        
        gesture_center = last_hand.center
        min_distance = float('inf')
        closest_face = None
        
        for face in face_results:
            face_x, face_y, face_w, face_h = face['bbox']
            face_center = (face_x + face_w // 2, face_y + face_h // 2)
            distance = np.sqrt((gesture_center[0] - face_center[0])**2 + 
                             (gesture_center[1] - face_center[1])**2)
            
            if distance < 200 and distance < min_distance:
                min_distance = distance
                closest_face = face
        
        return closest_face
    
    def draw_annotations(self, frame, results, debug_mode=True):
        annotated_frame = frame.copy()
        
        for face in results['faces']:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, name, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if debug_mode:
                confidence_text = f"{confidence:.1f}%"
                cv2.putText(annotated_frame, confidence_text, (x + 5, y + h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        if debug_mode and results['gestures']['bboxes'] is not None:
            bboxes = results['gestures']['bboxes'].astype(np.int32)
            ids = results['gestures']['ids']
            labels = results['gestures']['labels']
            
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
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
            
            if associated_face:
                text = f"{associated_face['name']}: {gesture_name}"
                color = (0, 255, 255)
            else:
                text = f"{gesture_name} gesture detected!"
                color = (255, 255, 255)
            
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        return annotated_frame


def run_capture_demo(args):
    """Run the demo with image capture capability."""
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    system = CombinedFaceGestureRecognizer(
        gesture_detector_path=args.detector,
        gesture_classifier_path=args.classifier
    )
    
    # Create output directory
    os.makedirs('report_images', exist_ok=True)
    
    print("\n" + "="*60)
    print("REPORT IMAGE CAPTURE MODE")
    print("="*60)
    print("\nControls:")
    print("  1 - Capture: Single person with gesture")
    print("  2 - Capture: Multi-person scenario")
    print("  3 - Capture: Gesture examples")
    print("  4 - Capture: Detection with bounding boxes")
    print("  SPACE - Quick capture (generic)")
    print("  's' - Toggle debug mode")
    print("  'q' or ESC - Quit")
    print("="*60)
    print("\nImages will be saved to: report_images/")
    print()
    
    debug_mode = True
    capture_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        start_time = time.time()
        
        results = system.process_frame(frame)
        annotated_frame = system.draw_annotations(frame, results, debug_mode=debug_mode)
        
        if debug_mode:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add capture instructions overlay
        cv2.putText(annotated_frame, "Press 1-4 to capture specific scenarios", 
                   (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
        
        cv2.imshow("Combined Face & Gesture Recognition - CAPTURE MODE", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('1'):
            filename = f"report_images/detection_example.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SAVED] {filename} (Single person with gesture)")
            capture_count += 1
        elif key == ord('2'):
            filename = f"report_images/multi_person.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SAVED] {filename} (Multi-person scenario)")
            capture_count += 1
        elif key == ord('3'):
            filename = f"report_images/gesture_examples.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SAVED] {filename} (Gesture examples)")
            capture_count += 1
        elif key == ord('4'):
            filename = f"report_images/bounding_boxes.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SAVED] {filename} (Detection with bounding boxes)")
            capture_count += 1
        elif key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_images/capture_{timestamp}.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SAVED] {filename}")
            capture_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"Capture session complete! Total images saved: {capture_count}")
    print(f"Images location: report_images/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images for report")
    
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
    
    args = parser.parse_args()
    run_capture_demo(args)
