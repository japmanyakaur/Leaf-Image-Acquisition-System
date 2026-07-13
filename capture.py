"""
Intelligent Vision-Based Leaf Image Acquisition System
--------------------------------------------------------
Runs on a fixed-focus webcam (e.g. Lenovo 300 FHD).

1. FULLY AUTOMATIC by default - exposure, brightness, and sensor gain are
   continuously adjusted by the system itself, in any environment (indoor
   or outdoor), with no setup. Leaf detection works on any background.
   Exposure is driven off the LEAF's own brightness (not the background),
   so a bright or dark background can't push the leaf itself over- or
   under-exposed. Sharpness/vein "good enough" targets are learned live
   from what the camera actually sees as you move the leaf around.

2. MANUAL OVERRIDE (secondary) - available two ways now:
     a) Keys i/k (exposure +/-) and o/l (brightness +/-) - a quick nudge
        that pauses automatic adjustment temporarily; it resumes on its
        own after a short idle period, exactly as before.
     b) A "Controls" side panel (a second window docked next to the
        preview) with sliders for Exposure, Brightness and Zoom, plus an
        "Auto Mode" toggle. Flipping the toggle to Manual (0) is a
        *persistent* lock - the sliders take direct control and it will
        NOT silently resume automatic like the key-nudges do. Flip it
        back to Auto (1) to hand control back to the automatic loop.

You'll see a bounding box drawn around the detected leaf on every frame
it is found, a green "CAPTURED" flash on screen, and a running counter
whenever a frame is saved.

Press 'm' to toggle a small debug inset showing the raw detection mask -
useful if the bounding box ever seems to disappear, since it shows
exactly what the detector is (or isn't) seeing.

USAGE
    python leaf_capture_system.py

CONTROLS (while a window is focused)
    q      - quit
    s      - force-save the current frame right now
    i / k  - manually raise / lower exposure  (temporary override)
    o / l  - manually raise / lower brightness (temporary override)
    m      - toggle debug mask inset (helps diagnose "no bounding box")
    Controls window sliders:
        Exposure       - direct exposure control (only takes effect
                          while Auto Mode = 0)
        Brightness     - direct brightness control (only takes effect
                          while Auto Mode = 0)
        Zoom           - digital zoom, 100 = 1.0x ... 300 = 3.0x
        Auto Mode      - 1 = automatic exposure (default), 0 = manual
                          (locks control to the Exposure/Brightness
                          sliders until switched back to 1)

OUTPUT
    Captured frames go to ./captures/leaf_YYYYMMDD_HHMMSS_<tag>_scoreXXX.jpg
    Saved frames are cropped tightly around the detected leaf ONLY - the
    rest of the frame is discarded before writing to disk.
    tag = auto (normal capture), manual (you pressed 's'), or
    timeout (guarantee-save fired before a fully sharp frame was seen -
    check these, they may be soft).
"""

import cv2
import numpy as np
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    camera_index: int = 1
    frame_width: int = 1920
    frame_height: int = 1080
    output_dir: str = "captures"

    # --- Leaf size relative to frame (area_ratio = leaf_area / frame_area) ---
    min_area_ratio: float = 0.02     # below this -> leaf reads as "too far"
    max_area_ratio: float = 0.55     # above this -> leaf reads as "too close"

    # --- Brightness (mean of grayscale, 0-255), measured over the leaf ---
    min_brightness: float = 70.0
    max_brightness: float = 205.0

    # --- Generic fallback / hard floor. Adaptive thresholds are never
    # allowed to relax below these, so a "learned" target can't end up
    # laxer than a sane baseline. ---
    sharpness_threshold: float = 90.0
    vein_score_threshold: float = 12.0

    # --- Stability / autocapture behaviour ---
    stability_frames_required: int = 15   # ~0.5s of "all checks pass"
    motion_threshold: float = 4.0         # fallback only, before adaptive learns the real noise floor
    capture_cooldown_sec: float = 3.0
    capture_timeout_sec: float = 9.0      # guarantee: save best-seen shot if "perfect" never hits this long

    # --- HSV range for green leaf segmentation. Kept fairly narrow to
    # TRUE greens on purpose: hue values below ~25 overlap skin tone, so a
    # wide lower bound causes the detector to lock onto a hand/arm holding
    # the leaf instead of the leaf itself. This is now only a secondary
    # check (see detect_leaf) - the primary segmentation is excess-green,
    # which doesn't depend on a hand-tuned hue cutoff at all. ---
    hsv_lower: tuple = (28, 30, 25)
    hsv_upper: tuple = (95, 255, 255)

    # --- Output framing / feedback ---
    crop_margin: float = 0.12          # tight crop around the leaf when saving (fraction of bbox size)
    capture_flash_sec: float = 1.8     # how long the on-screen "CAPTURED" flash stays visible
    low_confidence_score: float = 55.0 # below this, a timeout-save is flagged as possibly soft

    # --- Digital zoom (applied before detection, so bbox/crop stay consistent) ---
    zoom_factor: float = 1.0
    zoom_min: float = 1.0
    zoom_max: float = 3.0

    # --- Debug ---
    show_debug_mask: bool = False


# --------------------------------------------------------------------------- #
# Leaf detection - works on any background
# --------------------------------------------------------------------------- #

def estimate_background_color(frame, margin=50):
    h, w = frame.shape[:2]
    patches = [
        frame[0:margin, 0:margin],
        frame[0:margin, w - margin:w],
        frame[h - margin:h, 0:margin],
        frame[h - margin:h, w - margin:w],
        frame[0:margin, w // 2 - margin:w // 2 + margin],
    ]
    samples = np.vstack([p.reshape(-1, 3) for p in patches])
    return np.median(samples, axis=0)


def excess_green_mask(frame):
    """Excess Green Index: ExG = 2G - R - B, per pixel. This is the
    standard robust way to segment live green vegetation regardless of
    background, and - critically - it does NOT mistake a hand/arm for
    the leaf the way a hue-range threshold can: skin has R >= G, so its
    ExG value is low/negative and gets thresholded away, whereas an
    actual leaf's green channel dominates and its ExG is clearly
    positive. This is now the primary detector; the HSV hue mask below
    is only a secondary check.

    THRESHOLD BUG FIX: a plain global Otsu threshold on the ExG image
    picks whichever cutoff best separates the single MOST vividly-green
    region (e.g. a saturated green mesh/grille in the background) from
    everything else. If the real leaf is paler / less saturated than
    that other object, Otsu's cutoff lands ABOVE the leaf's ExG values -
    the leaf doesn't get merged with the background, it simply never
    becomes a foreground pixel at all, so it can never become a candidate
    contour downstream. To guard against that we cap the threshold at a
    fixed, more inclusive ceiling whenever Otsu would have picked
    something higher - this keeps paler green objects in the mask too,
    at the cost of possibly including some background as well. That's an
    acceptable trade because _best_valid_contour (below) then screens
    candidates on shape, solidity, AND color plausibility rather than
    blindly taking the largest blob."""
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_u8 = np.clip(exg, 0, 255).astype(np.uint8)
    exg_blur = cv2.GaussianBlur(exg_u8, (5, 5), 0)
    otsu_val, _ = cv2.threshold(exg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inclusive_thresh = min(otsu_val, 20.0)
    _, mask = cv2.threshold(exg_blur, inclusive_thresh, 255, cv2.THRESH_BINARY)
    return mask


def gray_world_white_balance(frame):
    """Corrects a global color cast (e.g. the cyan/teal tint seen from
    this webcam under mixed lighting, where white paper reads pale cyan
    instead of white) by scaling each channel so their means roughly
    match - the standard gray-world assumption. Run this FIRST, before
    anything ExG/HSV-based: an uncorrected cast pushes a neutral
    background toward "false green" and can mute or exaggerate the
    leaf's true greenness, which was quietly feeding both the mask and
    the color-plausibility check. Gains are clamped so a scene that is
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


def _mean_bgr_in_mask(frame, mask):
    if mask is None or not np.any(mask):
        return None
    pixels = frame[mask > 0].astype(np.float32)
    b, g, r = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()
    return b, g, r


def _is_plausible_leaf_color(frame, mask, margin=3.0):
    """Sanity check applied to WHATEVER contour a detector picked: the
    average color inside it must actually be green-dominant (G clearly
    above both R and B). This is what stops a skin-toned hand, wood grain,
    or a brownish background from ever being accepted as 'the leaf', even
    if it happened to pass the shape/solidity checks."""
    means = _mean_bgr_in_mask(frame, mask)
    if means is None:
        return False
    b, g, r = means
    return (g - r) > margin and (g - b) > margin


def background_diff_mask(frame, bg_color):
    diff = np.linalg.norm(frame.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _largest_valid_contour(frame, mask, frame_area, max_ratio=0.75, min_ratio=0.003, min_solidity=0.0):
    """Picks the best candidate contour, NOT simply the largest.

    Why: once excess_green_mask() was made more inclusive (see fix above),
    a small but very saturated background object (e.g. a green mesh)
    and a large but paler leaf can BOTH show up as separate contours in
    the same mask. Every candidate is scored on a combination of (a) how
    much of the frame it covers and (b) how clearly green-dominant its
    average color is (green minus red/blue). A candidate that isn't
    green-dominant at all is discarded outright, regardless of size - so
    a compact, highly-saturated patch can no longer beat out a bigger,
    correctly-toned leaf just by being "greener"."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    best_mask, best_bbox, best_area_ratio, best_score = None, None, 0.0, -1.0

    for c in contours:
        area = cv2.contourArea(c)
        area_ratio = area / frame_area
        if area_ratio < min_ratio or area_ratio > max_ratio:
            continue

        if min_solidity > 0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < min_solidity:
                continue

        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(candidate_mask, [c], -1, 255, thickness=cv2.FILLED)

        means = _mean_bgr_in_mask(frame, candidate_mask)
        if means is None:
            continue
        cb, cg, cr = means
        color_margin = min(cg - cr, cg - cb)
        if color_margin <= 3.0:
            continue  # not actually green-dominant - never let this win on size alone

        score = area_ratio * (1.0 + min(color_margin, 60.0) / 60.0)
        if score > best_score:
            best_score = score
            best_area_ratio = area_ratio
            x, y, w, h = cv2.boundingRect(c)
            best_bbox = (x, y, w, h)
            best_mask = candidate_mask

    if best_mask is None:
        return None, None, 0.0

    return best_mask, best_bbox, best_area_ratio


def detect_leaf(frame, cfg: Config):
    """Detection order, each gated by _is_plausible_leaf_color so a hand,
    wall, or wood surface can never be accepted just because it happened
    to pass a shape/solidity check:

    1. Excess-green (ExG) segmentation - the primary detector. Robust to
       background AND to skin tone, since it reasons about green-channel
       dominance rather than a fixed hue window.
    2. HSV true-green hue mask - secondary check, catches cases ExG
       misses (e.g. very low-saturation lighting).
    3. Background-difference - last resort for a badly backlit leaf,
       still gated by the same color-plausibility check afterward.

    Returns (mask_used_for_detection, bbox_or_None, area_ratio).
    mask_used_for_detection is always returned (even on failure) so the
    caller can show it in the debug inset."""
    frame_area = frame.shape[0] * frame.shape[1]

    exg_mask = excess_green_mask(frame)
    exg_result = _largest_valid_contour(frame, exg_mask, frame_area, max_ratio=0.75,
                                         min_ratio=0.003, min_solidity=0.30)
    if exg_result[1] is not None:
        return exg_result[0], exg_result[1], exg_result[2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    hue_mask = cv2.inRange(hsv, lower, upper)
    hue_result = _largest_valid_contour(frame, hue_mask, frame_area, max_ratio=0.75,
                                         min_ratio=0.003, min_solidity=0.30)
    if hue_result[1] is not None:
        return hue_result[0], hue_result[1], hue_result[2]

    bg_color = estimate_background_color(frame)
    bg_mask = background_diff_mask(frame, bg_color)
    bg_result = _largest_valid_contour(frame, bg_mask, frame_area, max_ratio=0.5,
                                        min_ratio=0.003, min_solidity=0.35)
    if bg_result[1] is not None:
        return bg_result[0], bg_result[1], bg_result[2]

    # Nothing passed validation - still hand back the combined raw masks
    # so the debug inset ('m' key) can show *why* nothing matched, instead
    # of a blank screen.
    combined = cv2.bitwise_or(exg_mask, cv2.bitwise_or(hue_mask, bg_mask))
    return combined, None, 0.0


# --------------------------------------------------------------------------- #
# Quality metrics - all restricted to leaf-only pixels via the mask, so
# background texture/brightness inside the bounding box can't distort them.
# --------------------------------------------------------------------------- #

def compute_brightness(gray_roi, mask=None):
    if mask is not None and np.any(mask):
        return float(np.mean(gray_roi[mask > 0]))
    return float(np.mean(gray_roi))


def compute_sharpness(gray_roi, mask=None):
    """Laplacian variance restricted to the leaf itself. Without the mask,
    sharp background texture inside the bounding box (table grain, a
    patterned surface) can make an out-of-focus leaf score as 'sharp' -
    masking it out fixes that."""
    if gray_roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    if mask is not None and np.any(mask):
        values = lap[mask > 0]
        if values.size == 0:
            return 0.0
        return float(values.var())
    return float(lap.var())


def compute_vein_score(gray_roi, leaf_mask_roi):
    if gray_roi.size == 0:
        return 0.0
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    edges = cv2.Canny(enhanced, 40, 120)
    if leaf_mask_roi is not None:
        edges = cv2.bitwise_and(edges, edges, mask=leaf_mask_roi)
        leaf_pixels = max(int(np.sum(leaf_mask_roi > 0)), 1)
    else:
        leaf_pixels = edges.size
    edge_pixels = int(np.sum(edges > 0))
    return float((edge_pixels / leaf_pixels) * 100.0)


# --------------------------------------------------------------------------- #
# Adaptive thresholds - learns "what good looks like" live, no button needed
# --------------------------------------------------------------------------- #

class AdaptiveThresholds:
    """This lens is fixed-focus, so there IS one real sharp distance you
    have to find by moving the leaf - no software removes that physical
    fact. But the system discovers that sweet spot itself from values it
    actually observes as you move the leaf, instead of you having to judge
    and confirm it. Starts from a generic safe fallback for the first
    couple of seconds, then switches to self-learned targets - but a
    learned target is never allowed to be laxer than the fallback."""

    def __init__(self, min_samples=20, max_samples=600,
                 sharp_fraction=0.85, vein_fraction=0.85,
                 motion_percentile=40, motion_margin=1.8):
        self.sharp_samples = []
        self.vein_samples = []
        self.motion_samples = []
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.sharp_fraction = sharp_fraction
        self.vein_fraction = vein_fraction
        self.motion_percentile = motion_percentile
        self.motion_margin = motion_margin

    def update(self, sharpness, vein_score, motion):
        self.sharp_samples.append(sharpness)
        self.vein_samples.append(vein_score)
        self.motion_samples.append(motion)
        for lst in (self.sharp_samples, self.vein_samples, self.motion_samples):
            if len(lst) > self.max_samples:
                del lst[:len(lst) - self.max_samples]

    @property
    def ready(self):
        return len(self.sharp_samples) >= self.min_samples

    def sharp_target(self, fallback):
        if not self.ready:
            return fallback
        learned = float(np.percentile(self.sharp_samples, 90)) * self.sharp_fraction
        return max(learned, fallback)

    def vein_target(self, fallback):
        if not self.ready:
            return fallback
        learned = float(np.percentile(self.vein_samples, 90)) * self.vein_fraction
        return max(learned, fallback)

    def motion_target(self, fallback):
        if len(self.motion_samples) < self.min_samples:
            return fallback
        return float(np.percentile(self.motion_samples, self.motion_percentile)) * self.motion_margin


# --------------------------------------------------------------------------- #
# Guidance
# --------------------------------------------------------------------------- #

class Guidance:
    NO_LEAF = "No leaf detected - place leaf in frame"
    MOVE_CLOSER = "Move Closer"
    MOVE_AWAY = "Move Away"
    INCREASE_LIGHT = "Increase Lighting"
    REDUCE_LIGHT = "Reduce Lighting / Avoid Glare"
    HOLD_STEADY = "Hold Steady - Focusing"
    ADJUST_FINE = "Adjust Slightly - Details Not Sharp"
    PERFECT = "Perfect Position"


def decide_guidance(cfg: Config, area_ratio, brightness, sharpness, vein_score,
                     leaf_found, sharp_thresh, vein_thresh):
    if not leaf_found:
        return Guidance.NO_LEAF, False
    if area_ratio < cfg.min_area_ratio:
        return Guidance.MOVE_CLOSER, False
    if area_ratio > cfg.max_area_ratio:
        return Guidance.MOVE_AWAY, False
    if brightness < cfg.min_brightness:
        return Guidance.INCREASE_LIGHT, False
    if brightness > cfg.max_brightness:
        return Guidance.REDUCE_LIGHT, False
    if sharpness < sharp_thresh:
        return Guidance.HOLD_STEADY, False
    if vein_score < vein_thresh:
        return Guidance.ADJUST_FINE, False
    return Guidance.PERFECT, True


def quality_score(cfg: Config, area_ratio, brightness, sharpness, vein_score):
    optimal_low = cfg.min_area_ratio + (cfg.max_area_ratio - cfg.min_area_ratio) * 0.25
    optimal_high = cfg.min_area_ratio + (cfg.max_area_ratio - cfg.min_area_ratio) * 0.65
    if optimal_low <= area_ratio <= optimal_high:
        pos_score = 1.0
    else:
        span = max(cfg.max_area_ratio - cfg.min_area_ratio, 1e-6)
        dist = min(abs(area_ratio - optimal_low), abs(area_ratio - optimal_high))
        pos_score = max(1.0 - dist / span, 0.0)

    bright_mid = (cfg.min_brightness + cfg.max_brightness) / 2
    bright_half_range = (cfg.max_brightness - cfg.min_brightness) / 2
    bright_score = max(1.0 - abs(brightness - bright_mid) / bright_half_range, 0.0)

    sharp_score = min(sharpness / (cfg.sharpness_threshold * 2.0), 1.0)
    vein_norm = min(vein_score / (cfg.vein_score_threshold * 2.0), 1.0)

    total = (0.35 * sharp_score + 0.30 * vein_norm +
             0.15 * bright_score + 0.20 * pos_score)
    return round(total * 100, 1)


# --------------------------------------------------------------------------- #
# Digital zoom - applied before detection so the bbox/crop stay consistent
# with everything downstream (nothing else needs to know zoom happened).
# --------------------------------------------------------------------------- #

def apply_digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.001:
        return frame
    h, w = frame.shape[:2]
    crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Camera setup and automatic exposure control
# --------------------------------------------------------------------------- #

def open_camera(cfg: Config):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg.camera_index)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    print("Camera property support (driver-dependent, -1 means unsupported):")
    for name, prop in [("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
                        ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
                        ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
                        ("GAIN", cv2.CAP_PROP_GAIN)]:
        print(f"  {name}: {cap.get(prop)}")

    return cap


def nudge_camera_property(cap, prop, delta, label, quiet=False):
    current = cap.get(prop)
    new_val = current + delta
    cap.set(prop, new_val)
    actual = cap.get(prop)
    if not quiet:
        print(f"  [{label}] requested {new_val:.2f}, driver reports {actual:.2f}")
    return actual


class AutoExposureController:
    """Drives the webcam toward a fixed, universal well-exposed brightness
    target - works the same indoors or outdoors, with zero setup. Uses
    exposure (shutter) as the primary lever, but switches to sensor GAIN
    once exposure is maxed out, since a longer shutter risks motion blur
    (which would hurt sharpness) while gain doesn't.

    Two ways to take manual control now:
      - Temporary: the i/k/o/l keys call pause_for_manual_override(), which
        pauses automatic adjustment and resumes it on its own after a
        short idle period (unchanged from before).
      - Persistent: the "Auto Mode" slider in the Controls window calls
        enable_manual_lock()/disable_manual_lock(). While locked, this
        controller does nothing at all and the Exposure/Brightness
        sliders drive the camera directly - it will NOT auto-resume until
        the toggle is flipped back.

    Two things that previously caused "too bright / randomly goes black":
    1. The camera's own FIRMWARE auto-exposure was never turned off, so it
       was fighting this software loop for control of the same knob -
       set_baseline() now explicitly forces the camera into manual mode.
    2. Adjustments had no ceiling, so a run of same-direction corrections
       could drift exposure/gain/brightness to an extreme (pure black or
       blown-out white) with nothing pulling it back. Every adjustment is
       now clamped to a bounded distance from a known-good baseline
       recorded at startup, and a near-black frame triggers an immediate
       reset to that baseline rather than continued drift.

    IMPORTANT: pass in the brightness measured over the LEAF region when
    one is visible, not the whole frame - otherwise a bright/dark
    background can pull exposure the wrong way for the leaf itself."""

    TARGET_LOW = 110.0
    TARGET_HIGH = 165.0
    TARGET_MID = (TARGET_LOW + TARGET_HIGH) / 2
    RESUME_AFTER_IDLE_SEC = 20.0

    def __init__(self, check_interval_frames: int = 10, max_step: float = 1.5,
                 exposure_max_drift: float = 6.0, gain_max_drift: float = 120.0,
                 brightness_max_drift: float = 80.0):
        self.check_interval = check_interval_frames
        self.max_step = max_step
        self.frame_count = 0
        self.paused = False
        self.resume_at = 0.0
        self.manual_lock = False
        self.last_status = "warming up"
        self.baseline_exposure = None
        self.baseline_gain = None
        self.baseline_brightness = None
        self.exposure_max_drift = exposure_max_drift
        self.gain_max_drift = gain_max_drift
        self.brightness_max_drift = brightness_max_drift
        self._black_frame_streak = 0

    def set_baseline(self, cap):
        """Call once after the camera has warmed up. Forces the camera out
        of its own auto-exposure mode and records the current
        exposure/gain/brightness as the safe anchor all later adjustments
        are bounded around."""
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow convention: 0.25 = manual
        if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) not in (0.25,):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 convention: 1 = manual
        print(f"  AUTO_EXPOSURE after manual-mode request: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

        self.baseline_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        self.baseline_gain = cap.get(cv2.CAP_PROP_GAIN)
        self.baseline_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        print(f"  baseline exposure={self.baseline_exposure:.2f}  "
              f"gain={self.baseline_gain:.2f}  brightness={self.baseline_brightness:.2f}")

    def _clamped_set(self, cap, prop, new_val, baseline, max_drift):
        if baseline is not None:
            low, high = baseline - max_drift, baseline + max_drift
            new_val = max(low, min(new_val, high))
        cap.set(prop, new_val)
        return cap.get(prop)

    def reset_to_baseline(self, cap):
        if self.baseline_exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.baseline_exposure)
            cap.set(cv2.CAP_PROP_GAIN, self.baseline_gain)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self.baseline_brightness)
            self.last_status = "reset to baseline (frame went black)"

    def pause_for_manual_override(self):
        """Temporary pause used by the i/k/o/l keys - resumes on its own."""
        self.paused = True
        self.resume_at = time.time() + self.RESUME_AFTER_IDLE_SEC

    def enable_manual_lock(self):
        """Persistent lock used by the Controls window 'Auto Mode' toggle -
        does NOT auto-resume; disable_manual_lock() must be called."""
        self.manual_lock = True
        self.last_status = "manual lock (Controls panel)"

    def disable_manual_lock(self):
        self.manual_lock = False

    def maybe_adjust(self, cap, mean_brightness: float):
        if self.manual_lock:
            # Fully hands-off: the Controls sliders are driving exposure/
            # brightness directly elsewhere in the main loop.
            return

        if self.paused:
            remaining = self.resume_at - time.time()
            if remaining <= 0:
                self.paused = False
            else:
                self.last_status = f"paused (manual override, resuming in {remaining:.0f}s)"
                return

        # Safety net: a run of consecutive near-black frames means whatever
        # direction we were correcting in overshot badly - snap straight
        # back to the known-good baseline instead of continuing to dig in.
        if mean_brightness < 8.0:
            self._black_frame_streak += 1
            if self._black_frame_streak >= 3:
                self.reset_to_baseline(cap)
                self._black_frame_streak = 0
            return
        else:
            self._black_frame_streak = 0

        self.frame_count += 1
        if self.frame_count % self.check_interval != 0:
            return

        mean = mean_brightness

        if self.TARGET_LOW <= mean <= self.TARGET_HIGH:
            self.last_status = f"stable (leaf mean {mean:.0f})"
            return

        error = self.TARGET_MID - mean
        direction = 1.0 if error > 0 else -1.0
        magnitude = min(abs(error) / 20.0, self.max_step)

        cur_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        applied_exp = self._clamped_set(cap, cv2.CAP_PROP_EXPOSURE, cur_exp + direction * magnitude,
                                         self.baseline_exposure, self.exposure_max_drift)
        exposure_hit_limit = abs(applied_exp - cur_exp) < 0.05

        if exposure_hit_limit and direction > 0:
            cur_gain = cap.get(cv2.CAP_PROP_GAIN)
            self._clamped_set(cap, cv2.CAP_PROP_GAIN, cur_gain + magnitude * 4,
                               self.baseline_gain, self.gain_max_drift)
        elif exposure_hit_limit and direction < 0:
            cur_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            self._clamped_set(cap, cv2.CAP_PROP_BRIGHTNESS, cur_bri - magnitude * 4,
                               self.baseline_brightness, self.brightness_max_drift)
        else:
            cur_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            self._clamped_set(cap, cv2.CAP_PROP_BRIGHTNESS, cur_bri + direction * magnitude * 2,
                               self.baseline_brightness, self.brightness_max_drift)

        word = "raising" if direction > 0 else "lowering"
        self.last_status = f"{word} (leaf mean {mean:.0f})"


def auto_gamma_correct(bgr, target_mean=140.0, tolerance=12.0):
    """Local shadow-lifting (CLAHE on L-channel) so a bright/uneven
    background doesn't hide a dark subject. The global gamma trim only
    kicks in when brightness is actually off target - previously it ran
    unconditionally every frame on top of the hardware exposure control,
    which double-corrected and amplified noise, hurting apparent
    sharpness/vein detail."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean > 1 and abs(mean - target_mean) > tolerance:
        gamma = np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(mean / 255.0 + 1e-6), 0.7, 1.5)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(result, table)
    return result


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def _beep():
    """Best-effort audible capture cue - silently does nothing if the
    platform doesn't support it, since the visual flash is the primary
    notification."""
    try:
        import winsound
        winsound.Beep(1000, 150)
    except Exception:
        try:
            print('\a', end='', flush=True)
        except Exception:
            pass


def save_frame(frame, cfg: Config, score: float, bbox=None, manual: bool = False,
                low_confidence: bool = False):
    """Crops tightly to the detected leaf (plus a small margin) before
    saving, instead of writing the whole 1080p frame. If bbox is None
    (no leaf was ever detected, e.g. a forced 's' save with nothing in
    frame) the full frame is saved as a fallback."""
    out = frame
    if bbox is not None:
        x, y, w, h = bbox
        mx, my = int(w * cfg.crop_margin), int(h * cfg.crop_margin)
        x1, y1 = max(x - mx, 0), max(y - my, 0)
        x2, y2 = min(x + w + mx, frame.shape[1]), min(y + h + my, frame.shape[0])
        if x2 > x1 and y2 > y1:
            out = frame[y1:y2, x1:x2]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "manual" if manual else ("timeout" if low_confidence else "auto")
    filename = f"leaf_{ts}_{tag}_score{int(round(score))}.jpg"
    path = os.path.join(cfg.output_dir, filename)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[CAPTURED] {path}  (quality score {score:.1f})")
    return path


# --------------------------------------------------------------------------- #
# Display - simplified overlay
# --------------------------------------------------------------------------- #

def rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


COL_PANEL = (32, 30, 28)
COL_SUCCESS = (110, 200, 80)
COL_WARNING = (25, 180, 235)
COL_DANGER = (70, 70, 225)
COL_TEXT_PRIMARY = (240, 240, 240)
COL_TEXT_MUTED = (160, 160, 160)
COL_BAR_BG = (46, 46, 52)
COL_BBOX = (0, 255, 60)


def status_color(ok):
    return COL_SUCCESS if ok else COL_DANGER


def draw_leaf_bbox(img, bbox):
    """Draws a clearly visible bounding box (thick outline + corner
    accents + label) around the detected leaf. Kept as its own function
    so it always runs whenever bbox is not None, regardless of anything
    else going on in the overlay."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    cv2.rectangle(img, (x, y), (x2, y2), COL_BBOX, 3)

    corner_len = max(min(w, h) // 6, 14)
    for cx, cy, dx, dy in [(x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), (255, 255, 255), 4)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), (255, 255, 255), 4)

    label = "LEAF"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ly = y - 10 if y - 10 - lh > 0 else y2 + lh + 10
    cv2.rectangle(img, (x, ly - lh - 6), (x + lw + 12, ly + 4), COL_BBOX, -1)
    cv2.putText(img, label, (x + 6, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_overlay(img, guidance_text, score, leaf_found, stable_count, stable_needed,
                  captures_count, flash_active):
    h, w = img.shape[:2]
    panel_x1, panel_y1, panel_x2, panel_y2 = 16, 16, w - 16, 126

    overlay = img.copy()
    rounded_rect(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), COL_PANEL, radius=16)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    if not leaf_found:
        accent = COL_DANGER
    elif guidance_text == Guidance.PERFECT:
        accent = COL_SUCCESS
    else:
        accent = COL_WARNING
    rounded_rect(img, (panel_x1, panel_y1), (panel_x1 + 6, panel_y2), accent, radius=3)

    cv2.putText(img, guidance_text, (panel_x1 + 28, panel_y1 + 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, accent, 2, cv2.LINE_AA)

    # score badge, top-right of panel
    badge_text = f"{score:.0f}"
    badge_cx, badge_cy = panel_x2 - 46, panel_y1 + 34
    cv2.circle(img, (badge_cx, badge_cy), 30, status_color(score >= 70), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.putText(img, badge_text, (badge_cx - tw // 2, badge_cy + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)

    # single progress bar: how close to a stable, auto-triggered capture
    bar_x1, bar_x2 = panel_x1 + 28, panel_x2 - 100
    bar_y = panel_y1 + 62
    rounded_rect(img, (bar_x1, bar_y), (bar_x2, bar_y + 14), COL_BAR_BG, radius=7)
    frac = 0.0 if stable_needed == 0 else min(stable_count / stable_needed, 1.0)
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * frac)
    if fill_x2 > bar_x1 + 14:
        rounded_rect(img, (bar_x1, bar_y), (fill_x2, bar_y + 14), status_color(frac >= 1.0), radius=7)
    cv2.putText(img, "holding steady", (bar_x1, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COL_TEXT_MUTED, 1, cv2.LINE_AA)

    cv2.putText(img, f"captured: {captures_count}", (w - 190, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT_MUTED, 1, cv2.LINE_AA)

    if flash_active:
        # Hard to miss: a full-frame color wash plus a labeled banner,
        # not just a thin border - the border alone was too easy to miss.
        wash = np.full_like(img, COL_SUCCESS)
        cv2.addWeighted(wash, 0.35, img, 0.65, 0, img)
        cv2.rectangle(img, (0, 0), (w, h), COL_SUCCESS, 22)

        msg = f"PHOTO CAPTURED  (#{captures_count})"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)
        tx, ty = (w - mw) // 2, h - 50
        cv2.rectangle(img, (tx - 20, ty - mh - 16), (tx + mw + 20, ty + 14), (0, 0, 0), -1)
        cv2.putText(img, msg, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, COL_SUCCESS, 3, cv2.LINE_AA)


def draw_debug_mask_inset(display, mask, leaf_found):
    """Small picture-in-picture of the raw detection mask, bottom-left
    corner. Toggle with 'm'. White = what the detector thinks is leaf.
    If the bounding box isn't appearing, this is the first place to look:
    an empty/noisy mask means the color/background thresholds need
    tuning for your lighting, not that the drawing code is broken."""
    h, w = display.shape[:2]
    inset_w, inset_h = 260, 150
    mask_small = cv2.resize(mask, (inset_w, inset_h))
    mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

    x1, y1 = 16, h - inset_h - 16
    x2, y2 = x1 + inset_w, y1 + inset_h
    border_color = COL_SUCCESS if leaf_found else COL_DANGER
    cv2.rectangle(display, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), border_color, 3)
    display[y1:y2, x1:x2] = mask_bgr
    cv2.putText(display, "detection mask ('m' to hide)", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT_PRIMARY, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Controls side panel - sliders for Exposure / Brightness / Zoom plus an
# Auto/Manual toggle. This is a second OpenCV window docked next to the
# preview (OpenCV trackbars must live in their own window - there's no
# native way to embed a slider inside an image window), positioned so it
# reads as a side panel rather than a separate, unrelated window.
# --------------------------------------------------------------------------- #

CONTROLS_WIN = "Controls"
SLIDER_EXPOSURE = "Exposure"
SLIDER_BRIGHTNESS = "Brightness"
SLIDER_ZOOM = "Zoom (x100)"
SLIDER_AUTO = "Auto(1)/Manual(0)"


def create_controls_window(cfg: Config, main_window_name: str):
    cv2.namedWindow(CONTROLS_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WIN, 340, 220)
    # Dock it to the left of the main preview window so it reads as a
    # side panel rather than a floating, unrelated window.
    cv2.moveWindow(main_window_name, 360, 30)
    cv2.moveWindow(CONTROLS_WIN, 10, 30)

    cv2.createTrackbar(SLIDER_EXPOSURE, CONTROLS_WIN, 100, 200, lambda v: None)
    cv2.createTrackbar(SLIDER_BRIGHTNESS, CONTROLS_WIN, 100, 200, lambda v: None)
    zoom_init = int(cfg.zoom_factor * 100)
    cv2.createTrackbar(SLIDER_ZOOM, CONTROLS_WIN, zoom_init, int(cfg.zoom_max * 100), lambda v: None)
    cv2.createTrackbar(SLIDER_AUTO, CONTROLS_WIN, 1, 1, lambda v: None)


def read_controls(cap, cfg: Config, exposure_ctrl: AutoExposureController):
    """Polls the Controls window sliders once per frame and applies them.
    Exposure/Brightness sliders only actually move the camera while
    Auto Mode is set to 0 (manual lock engaged) - otherwise the automatic
    controller owns those properties and the sliders are inert (moving
    them does nothing until you flip the toggle)."""
    auto_val = cv2.getTrackbarPos(SLIDER_AUTO, CONTROLS_WIN)
    if auto_val == 1 and exposure_ctrl.manual_lock:
        exposure_ctrl.disable_manual_lock()
    elif auto_val == 0 and not exposure_ctrl.manual_lock:
        exposure_ctrl.enable_manual_lock()

    if exposure_ctrl.manual_lock and exposure_ctrl.baseline_exposure is not None:
        exp_slider = cv2.getTrackbarPos(SLIDER_EXPOSURE, CONTROLS_WIN)
        bri_slider = cv2.getTrackbarPos(SLIDER_BRIGHTNESS, CONTROLS_WIN)

        exp_lo = exposure_ctrl.baseline_exposure - exposure_ctrl.exposure_max_drift
        exp_hi = exposure_ctrl.baseline_exposure + exposure_ctrl.exposure_max_drift
        target_exp = exp_lo + (exp_slider / 200.0) * (exp_hi - exp_lo)
        cap.set(cv2.CAP_PROP_EXPOSURE, target_exp)

        bri_lo = exposure_ctrl.baseline_brightness - exposure_ctrl.brightness_max_drift
        bri_hi = exposure_ctrl.baseline_brightness + exposure_ctrl.brightness_max_drift
        target_bri = bri_lo + (bri_slider / 200.0) * (bri_hi - bri_lo)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, target_bri)

    zoom_slider = cv2.getTrackbarPos(SLIDER_ZOOM, CONTROLS_WIN)
    cfg.zoom_factor = max(cfg.zoom_min, min(zoom_slider / 100.0, cfg.zoom_max))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cap = open_camera(cfg)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera_index in Config.")
        return

    print("Warming up camera...")
    for _ in range(15):
        cap.read()
        time.sleep(0.03)

    exposure_ctrl = AutoExposureController()
    exposure_ctrl.set_baseline(cap)
    adaptive = AdaptiveThresholds()

    window_name = "Leaf Capture System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    create_controls_window(cfg, window_name)

    stability_hist = deque(maxlen=cfg.stability_frames_required)
    prev_gray_full = None
    last_capture_time = 0.0
    flash_until = 0.0
    captures_count = 0

    leaf_present_since = None
    captured_this_presence = False
    presence_best_frame, presence_best_bbox, presence_best_score = None, None, -1.0

    last_status_print = 0.0

    print("Ready. Automatic exposure is running - point at a leaf, no setup needed.")
    print("Keys: q=quit  s=manual save  i/k=exposure +/-  o/l=brightness +/-  m=debug mask")
    print("Controls window: Exposure / Brightness / Zoom sliders + Auto/Manual toggle")

    while True:
        ok, raw_frame_full = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        read_controls(cap, cfg, exposure_ctrl)

        # Digital zoom is applied first so every downstream step (detection,
        # exposure sampling, gamma correction, cropping on save) operates on
        # the already-zoomed frame and stays geometrically consistent.
        raw_frame = apply_digital_zoom(raw_frame_full, cfg.zoom_factor)
        # Correct any global color cast (e.g. cyan/teal tint) before the
        # frame is used for anything else - detection, exposure sampling,
        # and gamma correction all assume roughly neutral color.
        raw_frame = gray_world_white_balance(raw_frame)

        raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        frame = auto_gamma_correct(raw_frame)
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        leaf_mask, bbox, area_ratio = detect_leaf(frame, cfg)
        leaf_found = bbox is not None

        # Exposure target: use the LEAF's own brightness (in the raw,
        # pre-gamma frame) when one is visible, whole-frame mean otherwise.
        if leaf_found:
            x, y, w, h = bbox
            x2b, y2b = min(x + w, raw_gray.shape[1]), min(y + h, raw_gray.shape[0])
            roi_raw = raw_gray[y:y2b, x:x2b]
            roi_mask_raw = leaf_mask[y:y2b, x:x2b]
            if roi_raw.size > 0 and np.any(roi_mask_raw):
                exposure_mean = float(np.mean(roi_raw[roi_mask_raw > 0]))
            else:
                exposure_mean = float(np.mean(raw_gray))
        else:
            exposure_mean = float(np.mean(raw_gray))
        exposure_ctrl.maybe_adjust(cap, exposure_mean)

        if leaf_found and prev_gray_full is not None:
            x, y, w, h = bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            cur_roi_m = gray_full[y:y2, x:x2]
            prev_roi_m = prev_gray_full[y:y2, x:x2]
            if cur_roi_m.shape == prev_roi_m.shape and cur_roi_m.size > 0:
                cur_norm = cur_roi_m.astype(np.float32) - float(np.mean(cur_roi_m))
                prev_norm = prev_roi_m.astype(np.float32) - float(np.mean(prev_roi_m))
                motion = float(np.mean(np.abs(cur_norm - prev_norm)))
            else:
                motion = 999.0
        else:
            motion = 999.0
        prev_gray_full = gray_full

        brightness = sharpness = vein_score = 0.0

        if leaf_found:
            x, y, w, h = bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            gray_roi = gray_full[y:y2, x:x2]
            roi_mask = leaf_mask[y:y2, x:x2]

            brightness = compute_brightness(gray_roi, roi_mask)
            sharpness = compute_sharpness(gray_roi, roi_mask)
            vein_score = compute_vein_score(gray_roi, roi_mask)

            # Bounding box is drawn unconditionally whenever a leaf is
            # found - a thick outline, corner accents, and a label, so it
            # can't be missed or blend into the background.
            draw_leaf_bbox(display, bbox)

            if cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio \
                    and cfg.min_brightness <= brightness <= cfg.max_brightness:
                adaptive.update(sharpness, vein_score, motion)

        sharp_thresh = adaptive.sharp_target(cfg.sharpness_threshold)
        vein_thresh = adaptive.vein_target(cfg.vein_score_threshold)
        live_motion_threshold = adaptive.motion_target(cfg.motion_threshold)

        guidance_text, all_pass = decide_guidance(
            cfg, area_ratio, brightness, sharpness, vein_score, leaf_found,
            sharp_thresh, vein_thresh)

        score = quality_score(cfg, area_ratio, brightness, sharpness, vein_score) if leaf_found else 0.0

        now = time.time()
        if leaf_found:
            if leaf_present_since is None:
                leaf_present_since = now
                captured_this_presence = False
                presence_best_frame, presence_best_bbox, presence_best_score = frame.copy(), bbox, score
            elif score > presence_best_score:
                presence_best_frame, presence_best_bbox, presence_best_score = frame.copy(), bbox, score
        else:
            leaf_present_since = None

        steady_now = all_pass and motion < live_motion_threshold
        stability_hist.append(steady_now)
        is_stable = len(stability_hist) == stability_hist.maxlen and all(stability_hist)

        if is_stable and (now - last_capture_time) > cfg.capture_cooldown_sec:
            # bbox here is the tight leaf box -> save_frame crops to just
            # the leaf (plus a small margin), discarding the rest of the
            # frame entirely.
            save_frame(frame, cfg, score, bbox=bbox)
            captures_count += 1
            _beep()
            last_capture_time = now
            flash_until = now + cfg.capture_flash_sec
            captured_this_presence = True
            stability_hist.clear()
        elif (leaf_present_since is not None and not captured_this_presence
              and (now - leaf_present_since) > cfg.capture_timeout_sec
              and presence_best_frame is not None):
            low_confidence = presence_best_score < cfg.low_confidence_score
            save_frame(presence_best_frame, cfg, presence_best_score,
                       bbox=presence_best_bbox, low_confidence=low_confidence)
            captures_count += 1
            _beep()
            if low_confidence:
                print("  note: best frame seen wasn't fully sharp - try holding the leaf a "
                      "touch more still, or move it slowly to help the system find focus")
            last_capture_time = now
            flash_until = now + cfg.capture_flash_sec
            captured_this_presence = True
            stability_hist.clear()

        if (now - last_status_print) > 1.0:
            ready_str = "learned" if adaptive.ready else f"learning ({len(adaptive.sharp_samples)}/{adaptive.min_samples})"
            mode_str = "MANUAL(locked)" if exposure_ctrl.manual_lock else (
                "paused(key)" if exposure_ctrl.paused else "auto")
            print(f"[status] guidance='{guidance_text}'  score={score:.1f}  "
                  f"sharp={sharpness:.0f}/{sharp_thresh:.0f}  vein={vein_score:.1f}/{vein_thresh:.1f}  "
                  f"adaptive={ready_str}  mode={mode_str}  zoom={cfg.zoom_factor:.2f}x  "
                  f"exposure={exposure_ctrl.last_status}")
            last_status_print = now

        draw_overlay(display, guidance_text, score, leaf_found,
                     len(stability_hist), stability_hist.maxlen,
                     captures_count, now < flash_until)

        if cfg.show_debug_mask:
            draw_debug_mask_inset(display, leaf_mask, leaf_found)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame, cfg, score, bbox=bbox, manual=True)
            captures_count += 1
            _beep()
            flash_until = time.time() + cfg.capture_flash_sec
        elif key == ord('m'):
            cfg.show_debug_mask = not cfg.show_debug_mask
        elif key == ord('i'):
            exposure_ctrl.pause_for_manual_override()
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, +1, "exposure +")
        elif key == ord('k'):
            exposure_ctrl.pause_for_manual_override()
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, -1, "exposure -")
        elif key == ord('o'):
            exposure_ctrl.pause_for_manual_override()
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, +10, "brightness +")
        elif key == ord('l'):
            exposure_ctrl.pause_for_manual_override()
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, -10, "brightness -")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()