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
    c  - calibration mode: hold a leaf at a KNOWN distance and press c,
         then type the distance in metres into the terminal. Do this at
         3+ different distances (e.g. 0.2, 0.5, 1.0, 2.0 m) across
         multiple runs. After 3+ points the script fits a physical
         area-vs-distance curve and gives you real distance estimates
         instead of a rough near/optimal/far heuristic.
    r  - reset the stability counter (if you got a false "hold steady")
    i / k - increase / decrease camera exposure (if the feed is too
            dark or too bright and CLAHE alone can't fix it)
    o / l - increase / decrease camera brightness

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

    # --- Startup exposure/brightness for this specific webcam+lighting ---
    # Found via live testing with the i/k/o/l keys: this driver's default
    # (EXPOSURE=-6) was too dark to detect anything. -1 is this driver's
    # ceiling (least-negative = longest shutter = brightest); 60 is near its
    # brightness ceiling too. Re-tune these two values if you change rooms,
    # lighting setup, or webcam - just watch the printed driver values while
    # pressing i/k/o/l and hardcode whatever worked.
    startup_exposure: float = -1.0
    startup_brightness: float = 60.0


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


def decide_guidance(cfg: Config, area_ratio, brightness, sharpness, vein_score, leaf_found):
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

    if sharpness < cfg.sharpness_threshold:
        return Guidance.HOLD_STEADY, False

    if vein_score < cfg.vein_score_threshold:
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

    # NOTE: we deliberately do NOT force CAP_PROP_AUTO_EXPOSURE here.
    # DirectShow's value convention for this property is inconsistent across
    # webcam drivers (some treat "3" as auto, others as a manual mode with a
    # very low fixed exposure) - forcing it can silently lock a camera into
    # near-black frames. Instead we apply known-good manual exposure/
    # brightness values (tuned live for this webcam - see Config) and let
    # the user nudge further with the i/k/o/l keys if lighting changes.
    cap.set(cv2.CAP_PROP_EXPOSURE, cfg.startup_exposure)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, cfg.startup_brightness)

    print("Camera property support (driver-dependent, -1 means unsupported):")
    for name, prop in [("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
                        ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
                        ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
                        ("GAIN", cv2.CAP_PROP_GAIN)]:
        print(f"  {name}: {cap.get(prop)}")

    return cap


def nudge_camera_property(cap, prop, delta, label):
    current = cap.get(prop)
    new_val = current + delta
    cap.set(prop, new_val)
    actual = cap.get(prop)
    print(f"  [{label}] requested {new_val:.2f}, driver reports {actual:.2f}")


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

    stability_hist = deque(maxlen=cfg.stability_frames_required)
    prev_gray_full = None
    last_capture_time = 0.0
    best_frame, best_score = None, -1.0
    session_captured = False

    print("Ready. Controls: q=quit  s=manual save  c=calibrate  r=reset stability  "
          "i/k=exposure +/-  o/l=brightness +/-")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        frame = auto_gamma_correct(frame)
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion estimate between consecutive frames (for "hold steady")
        if prev_gray_full is not None:
            motion = float(np.mean(cv2.absdiff(gray_full, prev_gray_full)))
        else:
            motion = 999.0
        prev_gray_full = gray_full

        leaf_mask, bbox, area_ratio = detect_leaf(frame, cfg)
        leaf_found = bbox is not None

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

        guidance_text, all_pass = decide_guidance(
            cfg, area_ratio, brightness, sharpness, vein_score, leaf_found)

        score = quality_score(cfg, area_ratio, brightness, sharpness, vein_score) if leaf_found else 0.0
        if score > best_score:
            best_score, best_frame = score, frame.copy()

        # Distance display: use calibrated estimate if available
        dist_estimate = calibrator.estimate(area_ratio) if leaf_found else None

        # Stability tracking for autocapture
        steady_now = all_pass and motion < cfg.motion_threshold
        stability_hist.append(steady_now)
        is_stable = len(stability_hist) == stability_hist.maxlen and all(stability_hist)

        now = time.time()
        if is_stable and (now - last_capture_time) > cfg.capture_cooldown_sec:
            save_frame(frame, cfg, score)
            last_capture_time = now
            session_captured = True
            stability_hist.clear()

        # ---- overlay ----
        draw_overlay(display, guidance_text, area_ratio, brightness, sharpness,
                     vein_score, score, dist_estimate, calibrator.is_calibrated,
                     leaf_found)

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
        elif key == ord('i'):
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, +1, "exposure +")
        elif key == ord('k'):
            nudge_camera_property(cap, cv2.CAP_PROP_EXPOSURE, -1, "exposure -")
        elif key == ord('o'):
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, +10, "brightness +")
        elif key == ord('l'):
            nudge_camera_property(cap, cv2.CAP_PROP_BRIGHTNESS, -10, "brightness -")

    cap.release()
    cv2.destroyAllWindows()

    if not session_captured and best_frame is not None:
        resp = input("No frame auto-captured this session. Save best frame "
                      f"(score {best_score:.1f}) now? [y/N]: ").strip().lower()
        if resp == "y":
            save_frame(best_frame, cfg, best_score, manual=True)


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
                  vein_score, score, dist_estimate, is_calibrated, leaf_found):
    h, w = img.shape[:2]
    panel_h = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    color = (60, 200, 60) if guidance_text == Guidance.PERFECT else (0, 165, 255)
    if not leaf_found:
        color = (60, 60, 220)

    cv2.putText(img, guidance_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    dist_str = f"{dist_estimate:.2f} m" if dist_estimate is not None else "not calibrated"
    lines = [
        f"Leaf area ratio: {area_ratio:.3f}   Est. distance: {dist_str}",
        f"Brightness: {brightness:.0f}   Sharpness: {sharpness:.0f}   Vein score: {vein_score:.1f}",
        f"Quality score: {score:.1f}/100" + ("" if is_calibrated else "   [press 'c' to calibrate distance]"),
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, 75 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 230, 230), 1)


if __name__ == "__main__":
    main()