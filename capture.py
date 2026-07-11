"""
Intelligent Vision-Based Leaf Image Acquisition System
--------------------------------------------------------
Runs on a fixed-focus webcam (e.g. Lenovo 300 FHD). Since the lens cannot
autofocus, this script continuously scores each live frame on leaf presence,
distance, brightness, sharpness and vein visibility, tells the user how to
move the leaf, and auto-captures the best frame once quality is high and
steady.

USAGE
    python leaf_capture_system.py

CONTROLS (while the preview window is focused)
    q  - quit (offers to save best frame from the session if none was
         auto-captured yet)
    s  - force-save the current frame right now
    t  - AUTO-TUNE: do this once per new environment/leaf setup. Hold a
         leaf where it looks good and press t; the system learns its own
         capture thresholds from what it sees. Auto-exposure (below) runs
         independently and doesn't need this step.
    c  - calibration mode (optional, cosmetic only): shows real distance
         in metres instead of just near/far. Not required for capture.
    r  - reset the stability counter (if you got a false "hold steady")
    a  - toggle automatic exposure on/off. It's ON by default and
         continuously adjusts exposure/gain/brightness by itself in ANY
         environment (indoor or outdoor) - no setup needed. Pressing any
         manual key below pauses it automatically; press 'a' to resume.
    i / k - manually raise / lower exposure (secondary - overrides auto)
    o / l - manually raise / lower brightness (secondary - overrides auto)
    g / h - manually raise / lower sensor gain (secondary - overrides auto)

OUTPUT
    Captured frames go to ./captures/leaf_YYYYMMDD_HHMMSS_scoreXXX.jpg
    Calibration data persists in ./distance_calibration.json
"""

import cv2
import numpy as np
import json
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

    calibration_file: str = "distance_calibration.json"
    output_dir: str = "captures"

    # --- Leaf size relative to frame (area_ratio = leaf_area / frame_area) ---
    # Used only as a FALLBACK before you've calibrated the camera. Tune these
    # two numbers against your own webcam/leaf sizes if capture feels wrong.
    min_area_ratio: float = 0.03     # below this -> leaf reads as "too far"
    max_area_ratio: float = 0.55     # above this -> leaf reads as "too close"
    optimal_area_low: float = 0.08
    optimal_area_high: float = 0.35

    # --- Brightness (mean of grayscale, 0-255) ---
    min_brightness: float = 70.0
    max_brightness: float = 205.0

    # --- Sharpness (variance of Laplacian on the leaf ROI) ---
    # Cheap fixed-focus webcams sit around 60-150 for "in focus enough".
    # Watch the on-screen number for a few seconds at a good distance and
    # re-tune this if capture never triggers or triggers too easily.
    sharpness_threshold: float = 90.0

    # --- Vein visibility (edge-density proxy, roughly 0-100 scale) ---
    vein_score_threshold: float = 12.0

    # --- Stability / autocapture behaviour ---
    stability_frames_required: int = 15   # ~0.5s at 30fps of "all checks pass"
    motion_threshold: float = 4.0         # mean abs frame-diff allowed while "steady"
    capture_cooldown_sec: float = 3.0

    # --- HSV range for green leaf segmentation (broad, override if needed) ---
    hsv_lower: tuple = (20, 25, 25)
    hsv_upper: tuple = (95, 255, 255)


# --------------------------------------------------------------------------- #
# Leaf detection
# --------------------------------------------------------------------------- #

def estimate_background_color(frame, margin=50):
    """Sample the four corners (+top-center) of the frame and take a median
    color. Assumes the leaf is roughly centered and doesn't touch all
    corners at once - true for a webcam-distance shot on a plain surface."""
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
    """Foreground = anything that looks meaningfully different from the
    sampled background color, regardless of what that color/brightness is.
    This is what makes detection work even when the leaf is underexposed
    into near-black - a dark silhouette is still very different from a
    bright uniform background."""
    diff = np.linalg.norm(frame.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _largest_valid_contour(mask, frame_area, max_ratio=0.85, min_ratio=0.003):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / frame_area

    # Reject noise specks and reject "whole frame flagged as foreground"
    # (happens if lighting is uneven enough to fool the background sample)
    if area_ratio < min_ratio or area_ratio > max_ratio:
        return None, None, 0.0

    leaf_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(largest)
    return leaf_mask, (x, y, w, h), area_ratio


def detect_leaf(frame, cfg: Config):
    """Hybrid segmentation: try green-hue detection AND background-difference
    detection, keep whichever finds the larger plausible blob. Hue detection
    works well for well-lit healthy-green leaves; background-difference
    works regardless of exposure or leaf color, which matters most when the
    leaf is backlit/underexposed into a silhouette. Returns
    (mask, bbox, area_ratio); bbox=None if nothing plausible was found.
    """
    frame_area = frame.shape[0] * frame.shape[1]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    hue_mask = cv2.inRange(hsv, lower, upper)
    hue_result = _largest_valid_contour(hue_mask, frame_area)

    bg_color = estimate_background_color(frame)
    bg_mask = background_diff_mask(frame, bg_color)
    bg_result = _largest_valid_contour(bg_mask, frame_area)

    candidates = [r for r in (hue_result, bg_result) if r[1] is not None]
    if not candidates:
        return hue_mask, None, 0.0

    # Prefer whichever method found a larger, more complete blob
    best = max(candidates, key=lambda r: r[2])
    return best


# --------------------------------------------------------------------------- #
# Quality metrics
# --------------------------------------------------------------------------- #

def compute_brightness(gray_roi):
    return float(np.mean(gray_roi))


def compute_sharpness(gray_roi):
    """Variance of Laplacian - higher = sharper. This is the metric you
    watch to find the fixed-focus 'sweet spot' distance: move the leaf
    slowly and this number will peak at the distance the lens is actually
    focused at."""
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


def compute_vein_score(gray_roi, leaf_mask_roi):
    """Proxy for fine-structure (vein/texture) visibility: density of
    strong edges inside the leaf region after local contrast enhancement.
    Not a calibrated physical unit, just a comparable relative score."""
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
    density = (edge_pixels / leaf_pixels) * 100.0
    return float(density)


# --------------------------------------------------------------------------- #
# Distance calibration (turns "area ratio" into an actual metres estimate)
# --------------------------------------------------------------------------- #

class DistanceCalibrator:
    """For a pinhole-camera approximation, apparent object area scales as
    1/distance^2, i.e. distance ~ k / sqrt(area_ratio). We fit k from a few
    (area_ratio, true_distance) samples the user provides via the 'c' key,
    using linear regression on x = 1/sqrt(area_ratio) vs y = distance."""

    def __init__(self, path: str):
        self.path = path
        self.samples = []  # list of (area_ratio, distance_m)
        self.k = None
        self.b = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.samples = [tuple(s) for s in data.get("samples", [])]
                self._fit()
            except (json.JSONDecodeError, OSError):
                self.samples = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"samples": self.samples}, f, indent=2)

    def add_sample(self, area_ratio: float, distance_m: float):
        if area_ratio <= 0:
            print("  [calibration] no leaf detected right now - not recorded.")
            return
        self.samples.append((area_ratio, distance_m))
        self._save()
        self._fit()
        print(f"  [calibration] recorded area_ratio={area_ratio:.4f} at {distance_m} m "
              f"({len(self.samples)} points total)")

    def _fit(self):
        if len(self.samples) < 3:
            self.k, self.b = None, None
            return
        xs = np.array([1.0 / np.sqrt(s[0]) for s in self.samples])
        ys = np.array([s[1] for s in self.samples])
        # y = k*x + b
        A = np.vstack([xs, np.ones_like(xs)]).T
        k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        self.k, self.b = float(k), float(b)

    def estimate(self, area_ratio: float):
        if self.k is None or area_ratio <= 0:
            return None
        return max(self.k / np.sqrt(area_ratio) + self.b, 0.0)

    @property
    def is_calibrated(self):
        return self.k is not None


# --------------------------------------------------------------------------- #
# Guidance engine
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


class AdaptiveThresholds:
    """Learns 'what good looks like' continuously from live use instead of
    requiring an explicit calibration step. This lens is fixed-focus, so
    there IS one real sharp distance you have to find by moving the leaf -
    no software removes that physical fact. But the system can discover
    that sweet spot itself from the values it actually observes as you
    naturally move the leaf around (which the on-screen guidance already
    encourages), rather than you having to judge and confirm it. Starts
    from a generic safe fallback for the first couple of seconds, then
    switches to self-learned targets - no button press required, though
    't' still exists as an optional fast-track if you want to skip the
    wait by confirming one good example."""

    def __init__(self, min_samples=20, max_samples=600,
                 sharp_fraction=0.80, vein_fraction=0.80,
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

    def seed(self, sharpness, vein_score, motion, repeats=15):
        """Optional fast-track: inject a user-confirmed good example
        several times so targets jump to it immediately instead of waiting
        to be organically discovered."""
        for _ in range(repeats):
            self.update(sharpness, vein_score, motion)

    @property
    def ready(self):
        return len(self.sharp_samples) >= self.min_samples

    def sharp_target(self, fallback):
        if not self.ready:
            return fallback
        return float(np.percentile(self.sharp_samples, 90)) * self.sharp_fraction

    def vein_target(self, fallback):
        if not self.ready:
            return fallback
        return float(np.percentile(self.vein_samples, 90)) * self.vein_fraction

    def motion_target(self, fallback):
        if len(self.motion_samples) < self.min_samples:
            return fallback
        # Low percentile = the calmer moments even within a session that
        # includes movement, i.e. an estimate of the real noise floor.
        return float(np.percentile(self.motion_samples, self.motion_percentile)) * self.motion_margin


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
    """0-100 combined score, used to pick the best frame of a session even
    if 'perfect' is never reached."""
    # Position score: 1.0 inside the optimal band, decaying outside it
    if cfg.optimal_area_low <= area_ratio <= cfg.optimal_area_high:
        pos_score = 1.0
    else:
        span = max(cfg.max_area_ratio - cfg.min_area_ratio, 1e-6)
        dist = min(abs(area_ratio - cfg.optimal_area_low),
                    abs(area_ratio - cfg.optimal_area_high))
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
# Camera setup helpers
# --------------------------------------------------------------------------- #

def open_camera(cfg: Config):
    # CAP_DSHOW is the reliable backend on Windows for UVC control properties
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg.camera_index)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    # NOTE: we deliberately do NOT force any startup exposure/brightness
    # values here, and do NOT force CAP_PROP_AUTO_EXPOSURE (DirectShow's
    # convention for it is inconsistent across drivers and forcing it can
    # silently lock a camera into a bad manual state). Instead,
    # AutoExposureController takes over immediately after the warm-up loop
    # and proportionally drives exposure/gain/brightness toward a good
    # target from whatever the driver happens to boot at - this is what
    # makes it work unmodified in any new environment, indoor or outdoor.
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
    """Drives the webcam toward a FIXED, universal well-exposed brightness
    target - not one derived from any per-room auto-tune - so it works the
    same whether you're indoors under a lamp or outdoors in daylight, with
    zero manual setup. This is the "automatic" mode; the i/k/o/l keys
    remain available any time you want to override it by hand.

    Design:
    - Checks the RAW incoming frame (before software CLAHE correction),
      since we're driving the actual sensor, not the display.
    - Adjustment size is PROPORTIONAL to how far off target the frame is,
      so it corrects a small day-to-day drift in a couple of steps but
      also recovers quickly from something drastic (e.g. walking from a
      dim room into full sun) instead of crawling there one unit at a time.
    - Uses exposure (shutter) as the primary lever, but when exposure is
      already at the driver's limit and more brightness is still needed,
      switches to sensor GAIN instead. Gain adds brightness without
      lengthening the shutter, so it doesn't add motion-blur risk -
      important since a longer shutter directly hurts the vein/sharpness
      score this whole system is judged on. Gain trades a little sensor
      noise for that safety, which is the right trade for a static leaf
      shot. BRIGHTNESS is used as a smaller final trim.
    """

    TARGET_LOW = 110.0
    TARGET_HIGH = 165.0
    TARGET_MID = (TARGET_LOW + TARGET_HIGH) / 2

    def __init__(self, check_interval_frames: int = 6, max_step: float = 3.0):
        self.check_interval = check_interval_frames
        self.max_step = max_step
        self.frame_count = 0
        self.enabled = True
        self.last_status = "auto-exposure: warming up"

    def maybe_adjust(self, cap, raw_gray):
        if not self.enabled:
            self.last_status = "auto-exposure: PAUSED (manual mode - press 'a' to resume)"
            return

        self.frame_count += 1
        if self.frame_count % self.check_interval != 0:
            return

        mean = float(np.mean(raw_gray))

        if self.TARGET_LOW <= mean <= self.TARGET_HIGH:
            self.last_status = f"auto-exposure: stable (raw mean {mean:.0f})"
            return

        error = self.TARGET_MID - mean          # positive => too dark
        direction = 1.0 if error > 0 else -1.0
        magnitude = min(abs(error) / 15.0, self.max_step)  # proportional, capped

        cur_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp + direction * magnitude)
        applied_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        exposure_hit_limit = abs(applied_exp - cur_exp) < 0.05

        applied_gain = cap.get(cv2.CAP_PROP_GAIN)
        applied_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)

        if exposure_hit_limit and direction > 0:
            # Shutter can't get any longer - brighten via gain instead so
            # sharpness doesn't keep degrading.
            cur_gain = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, cur_gain + magnitude * 4)
            applied_gain = cap.get(cv2.CAP_PROP_GAIN)
        elif exposure_hit_limit and direction < 0:
            cur_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, cur_bri - magnitude * 4)
            applied_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        else:
            cur_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, cur_bri + direction * magnitude * 2)
            applied_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)

        word = "raising" if direction > 0 else "lowering"
        self.last_status = (f"auto-exposure: {word} (raw mean {mean:.0f} -> "
                             f"exp {applied_exp:.1f}, gain {applied_gain:.1f}, bri {applied_bri:.1f})")


def auto_gamma_correct(bgr, target_mean=140.0):
    """Local shadow-lifting instead of whole-frame gamma. A bright uniform
    background (common in this setup) skews the whole-frame average high
    even when the actual subject is dark/backlit, so a global gamma curve
    barely touches the shadows. CLAHE on the L-channel in LAB space boosts
    *local* contrast/brightness independently in dark and bright regions,
    which is what actually recovers detail from a silhouetted leaf."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Light global lift on top, in case the whole frame is genuinely dim
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean > 1:
        gamma = np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(mean / 255.0 + 1e-6), 0.6, 1.8)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(result, table)

    return result


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    cfg = Config()
    load_autotune(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    calibrator = DistanceCalibrator(cfg.calibration_file)

    cap = open_camera(cfg)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera_index in Config.")
        return

    print("Warming up camera (letting auto-exposure settle)...")
    for _ in range(15):
        cap.read()
        time.sleep(0.03)

    exposure_ctrl = AutoExposureController()
    adaptive = AdaptiveThresholds()

    stability_hist = deque(maxlen=cfg.stability_frames_required)
    prev_gray_full = None
    last_capture_time = 0.0
    best_frame, best_score = None, -1.0
    session_captured = False

    tuning_mode = False
    tuning_samples = []
    tuning_start = 0.0
    TUNING_DURATION_SEC = 1.5

    last_status_print = 0.0

    print("Ready. Automatic exposure is running - point at a leaf, no setup needed.")
    print("The system learns its own sharpness/detail/steadiness targets live as you use it -")
    print("just move the leaf slowly until guidance says 'Perfect Position'. No calibration required.")
    print("Controls: q=quit  s=manual save  t=optional fast-track (confirm current position as 'good')  "
          "a=toggle auto-exposure  i/k/o/l/g/h=manual exposure/brightness/gain  "
          "c=optional distance readout  r=reset stability")

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        exposure_ctrl.maybe_adjust(cap, raw_gray)  # continuous software auto-exposure

        frame = auto_gamma_correct(raw_frame)
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        leaf_mask, bbox, area_ratio = detect_leaf(frame, cfg)
        leaf_found = bbox is not None

        # Motion estimate for "hold steady" - restricted to the leaf's own
        # bounding box, and brightness-normalized (mean subtracted from
        # each side) before comparing. This matters because the
        # auto-exposure controller changes overall frame brightness while
        # it converges - without normalizing, that brightness shift alone
        # would read as "motion" and repeatedly reset the steady-frame
        # counter even though the leaf never moved.
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
        roi_mask = None

        if leaf_found:
            x, y, w, h = bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            gray_roi = gray_full[y:y2, x:x2]
            roi_mask = leaf_mask[y:y2, x:x2]

            brightness = compute_brightness(gray_roi)
            sharpness = compute_sharpness(gray_roi)
            vein_score = compute_vein_score(gray_roi, roi_mask)

            cv2.rectangle(display, (x, y), (x2, y2), (60, 200, 60), 2)

            # Feed the adaptive learner - only from frames where the leaf is
            # a plausible size and reasonably lit, so a stray bad frame
            # doesn't skew what "good" means. Not while in tuning fast-track
            # (that has its own short focused burst below).
            if (not tuning_mode and cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio
                    and cfg.min_brightness <= brightness <= cfg.max_brightness):
                adaptive.update(sharpness, vein_score, motion)

        sharp_thresh = adaptive.sharp_target(cfg.sharpness_threshold)
        vein_thresh = adaptive.vein_target(cfg.vein_score_threshold)
        live_motion_threshold = adaptive.motion_target(cfg.motion_threshold)

        guidance_text, all_pass = decide_guidance(
            cfg, area_ratio, brightness, sharpness, vein_score, leaf_found,
            sharp_thresh, vein_thresh)

        score = quality_score(cfg, area_ratio, brightness, sharpness, vein_score) if leaf_found else 0.0
        if score > best_score:
            best_score, best_frame = score, frame.copy()

        # Distance display: use calibrated estimate if available
        dist_estimate = calibrator.estimate(area_ratio) if leaf_found else None

        # ---- optional fast-track sample collection ----
        if tuning_mode:
            if leaf_found:
                tuning_samples.append((area_ratio, brightness, sharpness, vein_score, motion))
            if time.time() - tuning_start >= TUNING_DURATION_SEC:
                tuning_mode = False
                finish_autotune(cfg, adaptive, tuning_samples)
                tuning_samples = []
                stability_hist.clear()

        # Stability tracking for autocapture (skip while tuning)
        steady_now = all_pass and motion < live_motion_threshold
        stability_hist.append(steady_now)
        is_stable = (not tuning_mode and len(stability_hist) == stability_hist.maxlen
                     and all(stability_hist))

        now = time.time()
        if is_stable and (now - last_capture_time) > cfg.capture_cooldown_sec:
            save_frame(frame, cfg, score)
            last_capture_time = now
            session_captured = True
            stability_hist.clear()

        if not tuning_mode and (now - last_status_print) > 1.0:
            ready_str = "learned" if adaptive.ready else f"learning ({len(adaptive.sharp_samples)}/{adaptive.min_samples})"
            print(f"[status] guidance='{guidance_text}'  motion={motion:.2f}/{live_motion_threshold:.2f}  "
                  f"steady={len(stability_hist)}/{stability_hist.maxlen}  score={score:.1f}  "
                  f"adaptive_targets={ready_str}")
            last_status_print = now

        # ---- overlay ----
        if tuning_mode:
            draw_tuning_overlay(display, leaf_found, len(tuning_samples))
        else:
            draw_overlay(display, guidance_text, area_ratio, brightness, sharpness,
                         vein_score, score, dist_estimate, calibrator.is_calibrated,
                         leaf_found, motion, live_motion_threshold,
                         len(stability_hist), stability_hist.maxlen,
                         exposure_ctrl.last_status)

        cv2.imshow("Leaf Capture System", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame, cfg, score, manual=True)
            session_captured = True
        elif key == ord('r'):
            stability_hist.clear()
        elif key == ord('c'):
            handle_calibration(calibrator, area_ratio)
        elif key == ord('t'):
            if leaf_found:
                tuning_mode = True
                tuning_samples = []
                tuning_start = time.time()
                print("[auto-tune] hold the leaf steady in its best position...")
            else:
                print("[auto-tune] place a leaf in frame first, then press t.")
        elif key == ord('i'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, +1, "exposure +")
        elif key == ord('k'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, -1, "exposure -")
        elif key == ord('o'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, +10, "brightness +")
        elif key == ord('l'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, -10, "brightness -")
        elif key == ord('g'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_GAIN, +10, "gain +")
        elif key == ord('h'):
            exposure_ctrl.enabled = False
            nudge_camera_property(cap, cv2.CAP_PROP_GAIN, -10, "gain -")
        elif key == ord('a'):
            exposure_ctrl.enabled = not exposure_ctrl.enabled
            state = "ENABLED (automatic)" if exposure_ctrl.enabled else "PAUSED (manual)"
            print(f"  [auto-exposure] {state}")

    cap.release()
    cv2.destroyAllWindows()

    if not session_captured and best_frame is not None:
        resp = input("No frame auto-captured this session. Save best frame "
                      f"(score {best_score:.1f}) now? [y/N]: ").strip().lower()
        if resp == "y":
            save_frame(best_frame, cfg, best_score, manual=True)


def load_autotune(cfg: Config, path: str = "autotune.json"):
    """If a previous auto-tune run saved thresholds, load them over the
    Config defaults so the system starts already configured for your
    webcam/room/leaf setup."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        print(f"Loaded auto-tuned thresholds from {path}.")
    except (json.JSONDecodeError, OSError):
        print(f"Could not read {path}, using default thresholds.")


def save_autotune(cfg: Config, path: str = "autotune.json"):
    data = {
        "sharpness_threshold": cfg.sharpness_threshold,
        "vein_score_threshold": cfg.vein_score_threshold,
        "min_area_ratio": cfg.min_area_ratio,
        "max_area_ratio": cfg.max_area_ratio,
        "optimal_area_low": cfg.optimal_area_low,
        "optimal_area_high": cfg.optimal_area_high,
        "min_brightness": cfg.min_brightness,
        "max_brightness": cfg.max_brightness,
        "motion_threshold": cfg.motion_threshold,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def finish_autotune(cfg: Config, adaptive: AdaptiveThresholds, samples):
    """Optional fast-track: take ~1.5s of live measurements from a leaf you
    confirmed looks good, and inject them into the adaptive learner so it
    is immediately 'ready' with a sensible target instead of waiting to
    discover one organically. This is a shortcut, not a requirement - the
    system learns the same targets on its own from normal use either way."""
    if len(samples) < 5:
        print("[auto-tune] not enough clean samples (leaf kept dropping out "
              "of detection) - try again, holding it steadier.")
        return

    areas = np.array([s[0] for s in samples])
    brights = np.array([s[1] for s in samples])
    sharps = np.array([s[2] for s in samples])
    veins = np.array([s[3] for s in samples])
    motions = np.array([s[4] for s in samples])

    med_area, med_bright = float(np.median(areas)), float(np.median(brights))
    med_sharp, med_vein, med_motion = float(np.median(sharps)), float(np.median(veins)), float(np.median(motions))

    # Widen the plausible area/brightness range around this example a bit -
    # convenience only, not required for sharpness/vein/motion learning.
    cfg.optimal_area_low = round(max(med_area * 0.7, 0.005), 4)
    cfg.optimal_area_high = round(min(med_area * 1.3, 0.9), 4)
    cfg.min_area_ratio = round(max(cfg.optimal_area_low * 0.6, 0.003), 4)
    cfg.max_area_ratio = round(min(cfg.optimal_area_high * 1.4, 0.9), 4)
    cfg.min_brightness = round(max(med_bright - 45, 20), 1)
    cfg.max_brightness = round(min(med_bright + 45, 250), 1)

    adaptive.seed(med_sharp, med_vein, med_motion, repeats=max(adaptive.min_samples + 5, 25))

    print(f"[fast-track] confirmed example applied from {len(samples)} samples - "
          f"adaptive targets are ready now instead of needing to be discovered "
          f"organically. The system will keep refining from here as you use it.")


def draw_tuning_overlay(img, leaf_found, sample_count):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    msg = "CONFIRMING POSITION - hold leaf steady..." if leaf_found else "CONFIRMING POSITION - leaf lost, repositioning..."
    color = (60, 200, 60) if leaf_found else (60, 60, 220)
    cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(img, f"samples collected: {sample_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)


def handle_calibration(calibrator: DistanceCalibrator, area_ratio: float):
    if area_ratio <= 0:
        print("  [calibration] no leaf detected - position the leaf first, then press c.")
        return
    try:
        raw = input("  Enter the TRUE distance from webcam to leaf, in metres: ").strip()
        distance_m = float(raw)
        calibrator.add_sample(area_ratio, distance_m)
    except ValueError:
        print("  [calibration] not a valid number, skipped.")


def save_frame(frame, cfg: Config, score: float, manual: bool = False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "manual" if manual else "auto"
    filename = f"leaf_{ts}_{tag}_score{int(round(score))}.jpg"
    path = os.path.join(cfg.output_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SAVED] {path}  (quality score {score:.1f})")


def draw_overlay(img, guidance_text, area_ratio, brightness, sharpness,
                  vein_score, score, dist_estimate, is_calibrated, leaf_found,
                  motion, motion_threshold, stable_count, stable_needed, exposure_status):
    h, w = img.shape[:2]
    panel_h = 180
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    color = (60, 200, 60) if guidance_text == Guidance.PERFECT else (0, 165, 255)
    if not leaf_found:
        color = (60, 60, 220)

    cv2.putText(img, guidance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    dist_str = f"{dist_estimate:.2f} m" if dist_estimate is not None else "not calibrated"
    motion_ok = motion < motion_threshold
    motion_color = (60, 200, 60) if motion_ok else (60, 60, 220)
    lines = [
        (f"Leaf area ratio: {area_ratio:.3f}   Est. distance: {dist_str}", (230, 230, 230)),
        (f"Brightness: {brightness:.0f}   Sharpness: {sharpness:.0f}   Vein score: {vein_score:.1f}", (230, 230, 230)),
        (f"Quality score: {score:.1f}/100" + ("" if is_calibrated else "   [press 'c' for distance, optional]"), (230, 230, 230)),
        (f"Motion: {motion:.1f} / {motion_threshold:.1f} (need below)   "
         f"Steady frames: {stable_count}/{stable_needed}", motion_color),
        (exposure_status, (180, 180, 180)),
    ]
    for i, (line, col) in enumerate(lines):
        cv2.putText(img, line, (20, 75 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, col, 1)


if __name__ == "__main__":
    main()