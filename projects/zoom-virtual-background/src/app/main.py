import argparse
import cv2 as cv
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from zoom_vbg.selfie_segmenter import SelfieSegmentation
from zoom_vbg.video_writer import VideoWriter
from zoom_vbg.backgrounds import load_backgrounds, preblur_background
from zoom_vbg.utils import set_capture_resolution, fit_to_frame, FPSMeter

PRESETS = {
    "fast":     {"blur": (3, 3), "alpha": 0.0, "edge": False, "morph": True,  "harm": False, "t0":0.40, "t1":0.70},
    "balanced": {"blur": (3, 3), "alpha": 0.6, "edge": True,  "morph": True,  "harm": False, "t0":0.40, "t1":0.70},
    "clean":    {"blur": (3, 3), "alpha": 0.8, "edge": True,  "morph": True,  "harm": True,  "t0":0.35, "t1":0.78},
}

def parse_args():
    ap = argparse.ArgumentParser(description="Zoom-style virtual background")
    ap.add_argument("--quality", choices=PRESETS.keys(), default="clean")
    ap.add_argument("--res", type=int, choices=[360,480,720,1080], default=720)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--bg", type=str, default="")
    ap.add_argument("--harmonize", choices=["on","off","auto"], default="auto")
    ap.add_argument("--record_dir", type=str, default=".")
    return ap.parse_args()

def main():
    args = parse_args()
    P = PRESETS[args.quality]

    # Open camera
    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    set_capture_resolution(cap, args.res)

    ok, first_frame = cap.read()
    if not ok or first_frame is None or first_frame.size == 0:
        raise RuntimeError("Could not read first frame")

    # Load backgrounds
    default_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "assets", "backgrounds"
    )
    backgrounds = load_backgrounds(default_folder)

    # Optional CLI-provided background
    selected_bg = None
    if args.bg and os.path.isfile(args.bg):
        tmp = cv.imread(args.bg)
        if tmp is not None:
            selected_bg = tmp

    background = selected_bg if selected_bg is not None else (backgrounds[0] if backgrounds else (255,255,255))
    if not isinstance(background, tuple):
        background = fit_to_frame(background, first_frame.shape)

    # Pre-blur once for quality look (no per-frame cost)
    pre_blurred_bg = background if isinstance(background, tuple) else cv.GaussianBlur(background, (5,5), 0)

    # Video writer
    writer = VideoWriter("SelfieSegmentation", out_dir=args.record_dir)

    # Segmenter
    seg = SelfieSegmentation(model=0)
    seg.alpha_ema = P["alpha"]
    seg.use_edge_refine = P["edge"]
    seg.use_morph = P["morph"]
    if args.harmonize == "on":
        seg.use_harmonize = True
    elif args.harmonize == "off":
        seg.use_harmonize = False
    else:
        seg.use_harmonize = P["harm"]
    seg.t0 = P["t0"]
    seg.t1 = P["t1"]
    seg.harmonize_strength = 0.40
    seg.use_gamma_on_fg = True
    seg.gamma_value = 0.92

    i = 0
    invis = False
    fps = FPSMeter()
    cv.namedWindow("Zoom VBG", cv.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cv.waitKey(5); continue

        frame = cv.GaussianBlur(frame, (3,3), 0)

        # Ensure bg matches current frame size (cams may renegotiate)
        if not isinstance(pre_blurred_bg, tuple) and pre_blurred_bg.shape[:2] != frame.shape[:2]:
            background = fit_to_frame(background, frame.shape)
            pre_blurred_bg = cv.GaussianBlur(background, (5,5), 0)

        bg_to_use = pre_blurred_bg

        out = seg.remove_background(frame, bg=bg_to_use, blur=P["blur"], invisible=invis, still_bg=first_frame)
        cv.imshow("Zoom VBG", out)
        writer.write(out)
        fps.tick()

        key = cv.waitKey(20) & 0xFF
        if key == ord('d'):
            if backgrounds and i < len(backgrounds)-1:
                i += 1
                background = fit_to_frame(backgrounds[i], frame.shape)
                pre_blurred_bg = background if isinstance(background, tuple) else cv.GaussianBlur(background, (5,5), 0)
                invis = False
                print(f"background index = {i}")
        elif key == ord('a'):
            if backgrounds and i > 0:
                i -= 1
                background = fit_to_frame(backgrounds[i], frame.shape)
                pre_blurred_bg = background if isinstance(background, tuple) else cv.GaussianBlur(background, (5,5), 0)
                invis = False
                print(f"background index = {i}")
        elif key == ord('s'):
            invis = True
        elif key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
