import cv2 as cv
import time

def set_capture_resolution(cap, target_h):
    w_map = {360:640, 480:854, 720:1280, 1080:1920}
    if target_h in w_map:
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  w_map[target_h])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_h)

def fit_to_frame(img, frame_shape):
    if img is None:
        return None
    H, W = frame_shape[:2]
    if img.shape[:2] != (H, W):
        img = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
    return img

class FPSMeter:
    def __init__(self):
        self.t = time.time()
        self.c = 0
    def tick(self):
        self.c += 1
        if self.c >= 60:
            now = time.time()
            fps = self.c / (now - self.t + 1e-6)
            print(f"FPS ~ {fps:.1f}")
            self.t = now
            self.c = 0
