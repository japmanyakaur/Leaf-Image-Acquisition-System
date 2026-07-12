"""
Intelligent Vision-Based Leaf Image Acquisition System
--------------------------------------------------------
Runs on a fixed-focus webcam (e.g. Lenovo 300 FHD). Two things happen, and
that's the whole system:

1. FULLY AUTOMATIC by default - exposure, brightness, and sensor gain are
   continuously adjusted by the system itself, in any environment (indoor
   or outdoor), with no setup. Leaf detection works on any background.
   Sharpness/detail "good enough" targets are learned live from what the
   camera actually sees as you move the leaf around (this lens is
   fixed-focus, so there is a real, physical sharp distance you have to
   find - no software removes that - but the system finds it statistically
   on its own instead of you having to judge and confirm it).

2. MANUAL OVERRIDE (secondary) - four keys let you directly nudge exposure
   and brightness if you ever want to. Using them pauses automatic
   adjustment temporarily; it resumes on its own after a short idle period.

USAGE
    python leaf_capture_system.py

CONTROLS (while the preview window is focused)
    q      - quit
    s      - force-save the current frame right now
    i / k  - manually raise / lower exposure
    o / l  - manually raise / lower brightness

OUTPUT
    Captured frames go to ./captures/leaf_YYYYMMDD_HHMMSS_scoreXXX.jpg
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

    # --- Brightness (mean of grayscale, 0-255) - generic, works with the
    # auto-exposure controller which keeps raw brightness near this anyway ---
    min_brightness: float = 70.0
    max_brightness: float = 205.0

    # --- Generic fallback used only for the first couple of seconds, before
    # the adaptive learner has enough samples to know your specific setup ---
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
# Quality metrics
# --------------------------------------------------------------------------- #

def compute_brightness(gray_roi):
    return float(np.mean(gray_roi))


def compute_sharpness(gray_roi):
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


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
    couple of seconds, then switches to self-learned targets."""

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
    period so automatic stays the default without needing a toggle key."""

    TARGET_LOW = 110.0
    TARGET_HIGH = 165.0
    TARGET_MID = (TARGET_LOW + TARGET_HIGH) / 2
    RESUME_AFTER_IDLE_SEC = 20.0

    def __init__(self, check_interval_frames: int = 6, max_step: float = 3.0):
        self.check_interval = check_interval_frames
        self.max_step = max_step
        self.frame_count = 0
        self.paused = False
        self.resume_at = 0.0
        self.last_status = "auto-exposure: warming up"

    def pause_for_manual_override(self):
        self.paused = True
        self.resume_at = time.time() + self.RESUME_AFTER_IDLE_SEC

    def maybe_adjust(self, cap, raw_gray):
        if self.paused:
            remaining = self.resume_at - time.time()
            if remaining <= 0:
                self.paused = False
            else:
                self.last_status = f"auto-exposure: paused (manual override, resuming in {remaining:.0f}s)"
                return

        self.frame_count += 1
        if self.frame_count % self.check_interval != 0:
            return

        mean = float(np.mean(raw_gray))

        if self.TARGET_LOW <= mean <= self.TARGET_HIGH:
            self.last_status = f"auto-exposure: stable (raw mean {mean:.0f})"
            return

        error = self.TARGET_MID - mean
        direction = 1.0 if error > 0 else -1.0
        magnitude = min(abs(error) / 15.0, self.max_step)

        cur_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp + direction * magnitude)
        applied_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        exposure_hit_limit = abs(applied_exp - cur_exp) < 0.05

        applied_gain = cap.get(cv2.CAP_PROP_GAIN)
        applied_bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)

        if exposure_hit_limit and direction > 0:
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
    """Local shadow-lifting (CLAHE on L-channel) so a bright/uneven
    background doesn't hide a dark subject, plus a light global gamma trim
    on top."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    if mean > 1:
        gamma = np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(mean / 255.0 + 1e-6), 0.6, 1.8)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(result, table)
    return result


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_frame(frame, cfg: Config, score: float, manual: bool = False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "manual" if manual else "auto"
    filename = f"leaf_{ts}_{tag}_score{int(round(score))}.jpg"
    path = os.path.join(cfg.output_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SAVED] {path}  (quality score {score:.1f})")


# --------------------------------------------------------------------------- #
# Display - clean dashboard overlay
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
COL_CHIP_BG_ON = (48, 66, 46)
COL_CHIP_BG_OFF = (46, 46, 52)


def status_color(ok):
    return COL_SUCCESS if ok else COL_DANGER


def draw_chip(img, x, y, w_chip, h_chip, label, ok):
    bg = COL_CHIP_BG_ON if ok else COL_CHIP_BG_OFF
    rounded_rect(img, (x, y), (x + w_chip, y + h_chip), bg, radius=8)
    cv2.circle(img, (x + 16, y + h_chip // 2), 5, status_color(ok), -1)
    cv2.putText(img, label, (x + 30, y + h_chip // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, COL_TEXT_PRIMARY, 1, cv2.LINE_AA)


def draw_overlay(img, guidance_text, area_ratio, brightness, sharpness,
                  vein_score, score, leaf_found, stable_count, stable_needed,
                  exposure_status, checks):
    w = img.shape[1]
    panel_x1, panel_y1, panel_x2, panel_y2 = 16, 16, w - 16, 232

    overlay = img.copy()
    rounded_rect(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), COL_PANEL, radius=16)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    if not leaf_found:
        accent = COL_DANGER
    elif guidance_text == Guidance.PERFECT:
        accent = COL_SUCCESS
    else:
        accent = COL_WARNING
    rounded_rect(img, (panel_x1, panel_y1), (panel_x1 + 6, panel_y2), accent, radius=3)

    cv2.putText(img, guidance_text, (panel_x1 + 28, panel_y1 + 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, accent, 2, cv2.LINE_AA)

    badge_text = f"{score:.0f}"
    badge_cx, badge_cy = panel_x2 - 46, panel_y1 + 34
    cv2.circle(img, (badge_cx, badge_cy), 30, status_color(score >= 70), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.putText(img, badge_text, (badge_cx - tw // 2, badge_cy + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, COL_TEXT_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(img, "score", (badge_cx - 18, badge_cy + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_MUTED, 1, cv2.LINE_AA)

    chip_y, chip_h = panel_y1 + 60, 30
    chip_w = (panel_x2 - panel_x1 - 28 - 3 * 10) // 4
    labels = [
        ("DISTANCE", checks["distance"]),
        ("LIGHT", checks["light"]),
        ("SHARPNESS", checks["sharp"]),
        ("DETAIL", checks["detail"]),
    ]
    cx = panel_x1 + 24
    for label, ok in labels:
        draw_chip(img, cx, chip_y, chip_w, chip_h, label, ok and leaf_found)
        cx += chip_w + 10

    bar_y = chip_y + chip_h + 20
    bar_x1, bar_x2 = panel_x1 + 24, panel_x2 - 24
    rounded_rect(img, (bar_x1, bar_y), (bar_x2, bar_y + 14), COL_CHIP_BG_OFF, radius=7)
    frac = 0.0 if stable_needed == 0 else min(stable_count / stable_needed, 1.0)
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * frac)
    if fill_x2 > bar_x1 + 14:
        rounded_rect(img, (bar_x1, bar_y), (fill_x2, bar_y + 14), status_color(frac >= 1.0), radius=7)
    cv2.putText(img, f"holding steady  {stable_count}/{stable_needed}", (bar_x1, bar_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT_MUTED, 1, cv2.LINE_AA)

    detail_line = (f"area {area_ratio:.3f}   brightness {brightness:.0f}   "
                    f"sharpness {sharpness:.0f}   vein {vein_score:.1f}")
    cv2.putText(img, detail_line, (bar_x1, bar_y + 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COL_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, exposure_status, (bar_x1, bar_y + 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COL_TEXT_MUTED, 1, cv2.LINE_AA)


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
    adaptive = AdaptiveThresholds()

    stability_hist = deque(maxlen=cfg.stability_frames_required)
    prev_gray_full = None
    last_capture_time = 0.0

    leaf_present_since = None
    captured_this_presence = False
    presence_best_frame, presence_best_score = None, -1.0

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
        exposure_ctrl.maybe_adjust(cap, raw_gray)

        frame = auto_gamma_correct(raw_frame)
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        leaf_mask, bbox, area_ratio = detect_leaf(frame, cfg)
        leaf_found = bbox is not None

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

            brightness = compute_brightness(gray_roi)
            sharpness = compute_sharpness(gray_roi)
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
                presence_best_frame, presence_best_score = frame.copy(), score
            elif score > presence_best_score:
                presence_best_frame, presence_best_score = frame.copy(), score
        else:
            leaf_present_since = None

        steady_now = all_pass and motion < live_motion_threshold
        stability_hist.append(steady_now)
        is_stable = len(stability_hist) == stability_hist.maxlen and all(stability_hist)

        if is_stable and (now - last_capture_time) > cfg.capture_cooldown_sec:
            save_frame(frame, cfg, score)
            last_capture_time = now
            captured_this_presence = True
            stability_hist.clear()
        elif (leaf_present_since is not None and not captured_this_presence
              and (now - leaf_present_since) > cfg.capture_timeout_sec
              and presence_best_frame is not None):
            save_frame(presence_best_frame, cfg, presence_best_score, manual=False)
            print(f"  (guaranteed-capture timeout at {cfg.capture_timeout_sec:.0f}s - "
                  f"saved best frame seen rather than waiting indefinitely)")
            last_capture_time = now
            captured_this_presence = True
            stability_hist.clear()

        if (now - last_status_print) > 1.0:
            ready_str = "learned" if adaptive.ready else f"learning ({len(adaptive.sharp_samples)}/{adaptive.min_samples})"
            print(f"[status] guidance='{guidance_text}'  motion={motion:.2f}/{live_motion_threshold:.2f}  "
                  f"steady={len(stability_hist)}/{stability_hist.maxlen}  score={score:.1f}  "
                  f"adaptive_targets={ready_str}")
            last_status_print = now

        checks = {
            "distance": leaf_found and cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio,
            "light": leaf_found and cfg.min_brightness <= brightness <= cfg.max_brightness,
            "sharp": leaf_found and sharpness >= sharp_thresh,
            "detail": leaf_found and vein_score >= vein_thresh,
        }
        draw_overlay(display, guidance_text, area_ratio, brightness, sharpness,
                     vein_score, score, leaf_found, len(stability_hist),
                     stability_hist.maxlen, exposure_ctrl.last_status, checks)

        cv2.imshow("Leaf Capture System", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame, cfg, score, manual=True)
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