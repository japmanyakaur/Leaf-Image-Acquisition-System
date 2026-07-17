"""Camera acquisition and per-frame image preparation: opening the device,
gamma/exposure-driven brightness correction, and color-cast correction.
Runs before detection/quality logic ever sees a frame."""

import os

import cv2
import numpy as np

from config import Config


def open_camera(cfg: Config):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg.camera_index)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    print("Camera property support (driver-dependent, -1 means unsupported):")
    for name, prop in [("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
                        ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
                        ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
                        ("GAIN", cv2.CAP_PROP_GAIN)]:
        print(f"  {name}: {cap.get(prop)}")

    return cap


def gray_world_white_balance(frame):
    """Corrects a global color cast (e.g. the cyan/teal tint seen from
    this webcam under mixed lighting, where white paper reads pale cyan
    instead of white) by scaling each channel so their means roughly
    match - the standard gray-world assumption. Run this FIRST, before
    anything ExG/HSV-based: an uncorrected cast pushes a neutral
    background toward "false green" and can mute or exaggerate the
    leaf's true greenness. Gains are clamped so a scene that is
    genuinely green overall doesn't get corrected away."""
    b, g, r = cv2.split(frame.astype(np.float32))
    mean_b, mean_g, mean_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    eps = 1e-3
    scale_b = np.clip(mean_gray / max(mean_b, eps), 0.6, 1.6)
    scale_g = np.clip(mean_gray / max(mean_g, eps), 0.6, 1.6)
    scale_r = np.clip(mean_gray / max(mean_r, eps), 0.6, 1.6)
    b = np.clip(b * scale_b, 0, 255)
    g = np.clip(g * scale_g, 0, 255)
    r = np.clip(r * scale_r, 0, 255)
    return cv2.merge((b, g, r)).astype(np.uint8)


def auto_gamma_correct(bgr, target_mean=140.0, tolerance=12.0, smoothed_mean=None):
    """Local shadow-lifting (CLAHE on L-channel) so a bright/uneven
    background doesn't hide a dark subject, followed by a LINEAR
    multiplicative brightness scale toward target_mean.

    A gamma/power-law curve has almost no effect near the ends of the
    0-255 range: a genuinely overexposed frame barely moves even with the
    correction pointed the right direction. A flat multiplicative scale
    has no such dead zone: it pulls EVERY pixel, including near-white
    ones, proportionally toward target. It can't recover pixels already
    fully clipped to 255, but it does pull the rest of the frame -
    including the leaf's own midtones - into a usable range.

    smoothed_mean, when given, is an EMA of this reading from prior
    frames and is what actually drives the correction strength - a
    single noisy or clutter-confused frame's raw mean can no longer yank
    it around and read as a brightness "flash". The function still
    returns this frame's raw (un-smoothed) mean as raw_mean so the
    caller can fold it into the EMA for the next call."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    raw_mean = float(np.mean(gray))
    mean = raw_mean if smoothed_mean is None else smoothed_mean
    if mean > 1 and abs(mean - target_mean) > tolerance:
        scale = np.clip(target_mean / mean, 0.3, 3.0)
        result = np.clip(result.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return result, raw_mean
