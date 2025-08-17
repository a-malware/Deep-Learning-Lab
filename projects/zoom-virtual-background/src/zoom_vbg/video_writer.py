import cv2 as cv
import datetime
import os

class VideoWriter:
    def __init__(self, name_root="SelfieSegmentation", out_dir="."):
        ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
        self.base = f"{name_root}-{ts}"
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{self.base}.mp4")
        self.writer = None
        self.fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.fps = 20.0
        print("Video Saving initialized")
        print(f"name = {self.path}")

    def write(self, frame):
        h, w = frame.shape[:2]
        if self.writer is None:
            self.writer = cv.VideoWriter(self.path, self.fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                print("Warning: VideoWriter failed to open; disabling recording.")
                self.writer = None
        if self.writer is not None:
            self.writer.write(frame)
