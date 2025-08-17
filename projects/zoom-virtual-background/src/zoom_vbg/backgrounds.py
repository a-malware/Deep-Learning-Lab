import os
import cv2 as cv
from .utils import fit_to_frame

def load_backgrounds(folder):
    bgs = []
    if os.path.isdir(folder):
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            img = cv.imread(path)
            if img is not None:
                bgs.append(img)
    return bgs

def preblur_background(bg, frame_shape, ksize=(5,5)):
    if isinstance(bg, tuple):
        return bg
    bg = fit_to_frame(bg, frame_shape)
    return cv.GaussianBlur(bg, ksize, 0)
