"""Quality metrics (sharpness, vein visibility, brightness), adaptive
"what good looks like" thresholds, guidance text, and the overall quality
score. All rule-based classical CV - no ML model."""

import cv2
import numpy as np

from config import Config


def compute_brightness(gray_roi, mask=None):
    if mask is not None and np.any(mask):
        return float(np.mean(gray_roi[mask > 0]))
    return float(np.mean(gray_roi))


def compute_sharpness(gray_roi, mask=None):
    """Laplacian variance restricted to the leaf itself. Without the mask,
    sharp background texture inside the bounding box (table grain, a
    patterned surface) can make an out-of-focus leaf score as 'sharp'."""
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


class AdaptiveThresholds:
    """This lens is fixed-focus, so there IS one real sharp distance you
    have to find by moving the leaf. The system discovers that sweet spot
    from values it actually observes as you move the leaf, rather than a
    hardcoded number. Starts from a generic safe fallback, then switches
    to self-learned targets - but a learned target is never allowed to be
    laxer than the fallback."""

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
