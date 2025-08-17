from typing import Optional, Tuple
import os
import cv2 as cv
import numpy as np
import mediapipe as mp

# Reduce verbose logs
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

Color = Tuple[int,int,int]

class SelfieSegmentation:
    def __init__(self, model:int=0):
        self.model = model
        self._mp_ss = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=model)
        # Quality-first defaults (override from runner if needed)
        self.prev_mask = None
        self.alpha_ema = 0.8
        self.use_morph = True
        self.use_edge_refine = True
        self.use_harmonize = True
        self.harmonize_strength = 0.40
        self.use_gamma_on_fg = True
        self.gamma_value = 0.92
        self.t0 = 0.35
        self.t1 = 0.78

    # ---------- helpers ----------
    def _ema(self, m):
        if self.alpha_ema <= 0:
            return m
        if self.prev_mask is None:
            self.prev_mask = m.copy()
            return m
        sm = self.alpha_ema*m + (1.0-self.alpha_ema)*self.prev_mask
        self.prev_mask = sm.copy()
        return sm

    def _morph(self, m, k=3):
        if not self.use_morph: return m
        m8 = (m*255).astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k,k))
        m8 = cv.morphologyEx(m8, cv.MORPH_OPEN, kernel, 1)
        m8 = cv.morphologyEx(m8, cv.MORPH_CLOSE, kernel, 1)
        return m8.astype(np.float32)/255.0

    def _edge_refine(self, m, edge_band=5, d=5, sigmaColor=70, sigmaSpace=3):
        if not self.use_edge_refine: return m
        m8 = (m*255).astype(np.uint8)
        edges = cv.Canny(m8, 50, 150)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (edge_band*2+1, edge_band*2+1))
        band = cv.dilate(edges, kernel, 1) > 0
        if not band.any(): return m
        m8_bi = cv.bilateralFilter(m8, d, sigmaColor, sigmaSpace)
        m8[band] = m8_bi[band]
        return m8.astype(np.float32)/255.0

    def _soft_alpha(self, m, t0=None, t1=None):
        t0 = self.t0 if t0 is None else t0
        t1 = self.t1 if t1 is None else t1
        denom = max(1e-6, (t1-t0))
        return np.clip((m - t0)/denom, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _gamma(img, gamma=0.92):
        lut = (np.clip(((np.arange(256)/255.0)**(1.0/gamma))*255.0,0,255)).astype(np.uint8)
        return cv.LUT(img, lut)

    def _harmonize(self, fg, bg, mask, strength=None):
        if not self.use_harmonize: return fg
        s = self.harmonize_strength if strength is None else strength
        if s <= 0: return fg
        m = (mask > 0.5)[...,None]
        try:
            fg_pix = fg[m].reshape(-1,3) if m.any() else fg.reshape(-1,3)
            bg_pix = bg.reshape(-1,3)
            fg_mean, fg_std = fg_pix.mean(0), fg_pix.std(0) + 1e-6
            bg_mean, bg_std = bg_pix.mean(0), bg_pix.std(0) + 1e-6
            gain = bg_std/fg_std
            bias = bg_mean - fg_mean*gain
            adj = np.clip(fg.astype(np.float32)*gain + bias, 0, 255).astype(np.uint8)
            return (fg.astype(np.float32)*(1-s) + adj.astype(np.float32)*s).astype(np.uint8)
        except Exception:
            return fg

    # ---------- API ----------
    def remove_background(self,
                          frame_bgr: np.ndarray,
                          bg: Optional[np.ndarray|Color]=(255,255,255),
                          blur=(3,3),
                          invisible=False,
                          still_bg: Optional[np.ndarray]=None) -> np.ndarray:
        # Flip + RGB
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = self._mp_ss.process(rgb)
        except Exception:
            return self._fallback(frame_bgr, bg)

        mask = getattr(res, "segmentation_mask", None)
        if mask is None or not isinstance(mask, np.ndarray):
            return self._fallback(frame_bgr, bg)

        m = mask.astype(np.float32)
        m = self._ema(m)
        m = self._morph(m, k=3)
        m = self._edge_refine(m, edge_band=5, d=5, sigmaColor=70, sigmaSpace=3)

        alpha = self._soft_alpha(m).astype(np.float32)
        if alpha.ndim != 2:
            alpha = alpha.squeeze()
        a3 = alpha[...,None]

        # Tuple background
        if isinstance(bg, tuple):
            canvas = np.zeros_like(frame_bgr); canvas[:] = bg
            out = (a3*frame_bgr.astype(np.float32) + (1.0-a3)*canvas.astype(np.float32)).astype(np.uint8)
            return out

        # Invisible mode
        if invisible:
            still = still_bg if still_bg is not None else frame_bgr
            still = cv.GaussianBlur(still, blur, 0)
            still = self._fit(still, frame_bgr.shape)
            inv = (1.0 - a3)
            return (inv*still.astype(np.float32) + a3*frame_bgr.astype(np.float32)).astype(np.uint8)

        # Image background
        if bg is None:
            return frame_bgr
        if bg.ndim == 2:
            bg = cv.cvtColor(bg, cv.COLOR_GRAY2BGR)
        elif bg.ndim == 3 and bg.shape[2] == 4:
            bg = cv.cvtColor(bg, cv.COLOR_BGRA2BGR)

        bg_blur = cv.GaussianBlur(bg, blur, 0)
        bg_blur = self._fit(bg_blur, frame_bgr.shape)

        # Optional gamma if BG bright
        if self.use_gamma_on_fg:
            Y = (0.114*bg_blur[...,0] + 0.587*bg_blur[...,1] + 0.299*bg_blur[...,2]).mean()
            if Y > 170:
                frame_bgr = self._gamma(frame_bgr, self.gamma_value)

        # Harmonize
        try:
            frame_bgr = self._harmonize(frame_bgr, bg_blur, m, self.harmonize_strength)
        except Exception:
            pass

        return (a3*frame_bgr.astype(np.float32) + (1.0-a3)*bg_blur.astype(np.float32)).astype(np.uint8)

    @staticmethod
    def _fit(img, frame_shape):
        H, W = frame_shape[:2]
        if img.shape[:2] != (H, W):
            img = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)
        return img

    @staticmethod
    def _fallback(frame_bgr, bg):
        if isinstance(bg, tuple):
            canvas = np.zeros_like(frame_bgr); canvas[:] = bg
            return canvas
        if bg is not None:
            H, W = frame_bgr.shape[:2]
            if bg.shape[:2] != (H, W):
                bg = cv.resize(bg, (W, H), interpolation=cv.INTER_AREA)
            return bg
        return frame_bgr
