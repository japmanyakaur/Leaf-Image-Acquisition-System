"""
Intelligent Vision-Based Leaf Image Acquisition System
--------------------------------------------------------
Runs on a fixed-focus webcam (e.g. Lenovo 300 FHD). Two things happen, and
that's the whole system:

1. FULLY AUTOMATIC by default - exposure, brightness, and sensor gain are
   continuously adjusted by the system itself, in any environment (indoor
   or outdoor), with no setup. Leaf detection works on any background.
   Exposure is driven off the LEAF's own brightness (not the background),
   so a bright or dark background can't push the leaf itself over- or
   under-exposed. Sharpness/vein "good enough" targets are learned live
   from what the camera actually sees as you move the leaf around (this
   lens is fixed-focus, so there is a real, physical sharp distance you
   have to find - no software removes that - but the system finds it
   statistically on its own instead of you having to judge and confirm it).

2. MANUAL OVERRIDE (secondary) - four keys let you directly nudge exposure
   and brightness if you ever want to. Using them pauses automatic
   adjustment temporarily; it resumes on its own after a short idle period.

You'll see a green "CAPTURED" flash on screen and a running counter
whenever a frame is saved, so you don't have to watch the console.

USAGE
    python leaf_capture_system.py

CONTROLS (while the preview window is focused)
    q      - quit
    s      - force-save the current frame right now
    i / k  - manually raise / lower exposure
    o / l  - manually raise / lower brightness

OUTPUT
    Captured frames go to ./captures/leaf_YYYYMMDD_HHMMSS_<tag>_scoreXXX.jpg
    Saved frames are cropped tightly around the detected leaf.
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
    min_area_ratio: float = 0.03     # below this -> leaf reads as "too far"
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

    # --- HSV range for green leaf segmentation (broad, override if needed) ---
    hsv_lower: tuple = (20, 25, 25)
    hsv_upper: tuple = (95, 255, 255)

    # --- Output framing / feedback ---
    crop_margin: float = 0.12          # tight crop around the leaf when saving (fraction of bbox size)
    capture_flash_sec: float = 1.8     # how long the on-screen "CAPTURED" flash stays visible
    low_confidence_score: float = 55.0 # below this, a timeout-save is flagged as possibly soft


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


def background_diff_mask(frame, bg_color):
    diff = np.linalg.norm(frame.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _largest_valid_contour(mask, frame_area, max_ratio=0.75, min_ratio=0.003, min_solidity=0.0):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    area_ratio = area / frame_area

    if area_ratio < min_ratio or area_ratio > max_ratio:
        return None, None, 0.0

    if min_solidity > 0:
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        if solidity < min_solidity:
            return None, None, 0.0

    leaf_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(largest)
    return leaf_mask, (x, y, w, h), area_ratio


def detect_leaf(frame, cfg: Config):
    """Leaf-color (green hue) detection is tried FIRST and used whenever it
    finds anything plausible - it works on any background because it's
    reasoning about the leaf's own color, not the scene behind it.
    Background-difference detection is only a fallback for when color
    detection finds nothing at all (e.g. a badly backlit leaf), and even
    then requires a solid, compact shape so real-world clutter can't be
    misread as one giant leaf."""
    frame_area = frame.shape[0] * frame.shape[1]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    hue_mask = cv2.inRange(hsv, lower, upper)
    hue_result = _largest_valid_contour(hue_mask, frame_area, max_ratio=0.75,
                                         min_ratio=0.003, min_solidity=0.35)
    if hue_result[1] is not None:
        return hue_result

    bg_color = estimate_background_color(frame)
    bg_mask = background_diff_mask(frame, bg_color)
    bg_result = _largest_valid_contour(bg_mask, frame_area, max_ratio=0.5,
                                        min_ratio=0.003, min_solidity=0.45)
    if bg_result[1] is not None:
        return bg_result

    return hue_mask, None, 0.0


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
    (which would hurt sharpness) while gain doesn't. Manual i/k/o/l keys
    pause this temporarily; it resumes automatically after a short idle
    period so automatic stays the default without needing a toggle key.

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
        self.paused = True
        self.resume_at = time.time() + self.RESUME_AFTER_IDLE_SEC

    def maybe_adjust(self, cap, mean_brightness: float):
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
    saving, instead of writing the whole 1080p frame."""
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


def status_color(ok):
    return COL_SUCCESS if ok else COL_DANGER


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
    print("Controls: q=quit  s=manual save  i/k=exposure +/-  o/l=brightness +/-")

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

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

            cv2.rectangle(display, (x, y), (x2, y2), (60, 200, 60), 2)

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
            print(f"[status] guidance='{guidance_text}'  score={score:.1f}  "
                  f"sharp={sharpness:.0f}/{sharp_thresh:.0f}  vein={vein_score:.1f}/{vein_thresh:.1f}  "
                  f"adaptive={ready_str}  exposure={exposure_ctrl.last_status}")
            last_status_print = now

        draw_overlay(display, guidance_text, score, leaf_found,
                     len(stability_hist), stability_hist.maxlen,
                     captures_count, now < flash_until)

        cv2.imshow("Leaf Capture System", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame, cfg, score, bbox=bbox, manual=True)
            captures_count += 1
            _beep()
            flash_until = time.time() + cfg.capture_flash_sec
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