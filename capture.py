"""
1. FULLY AUTOMATIC by default - exposure, brightness, and sensor gain are
   continuously adjusted by the system itself, in any environment (indoor
   or outdoor), with no setup. Leaf detection works on any background.
   Exposure is driven off the LEAF's own brightness (not the background),
   so a bright or dark background can't push the leaf itself over- or
   under-exposed. Sharpness/vein "good enough" targets are learned live
   from what the camera actually sees as you move the leaf around.

   AUTO-ZOOM: as soon as anything leaf-like is detected, the system pans
   and zooms TOWARD it - narrowing the field of view onto the leaf and
   cropping away background clutter (tripods, other objects) BEFORE
   relying on the detector for anything else. It keeps adjusting until the
   leaf is centered and at a good size, then stops and lets the normal
   detection/quality/capture pipeline run on that now-clean, zoomed-in
   view - the same as it always has. If the leaf is lost for a couple of
   seconds, it resets back to a wide view and searches again.

2. MANUAL OVERRIDE (secondary) - available two ways now:
     a) Keys i/k (exposure +/-) and o/l (brightness +/-) - a quick nudge
        that pauses automatic adjustment temporarily; it resumes on its
        own after a short idle period, exactly as before.
     b) A "Controls" side panel (a second window docked next to the
        preview) with sliders for Exposure, Brightness and Zoom, plus an
        "Auto Mode" toggle. Flipping the toggle to Manual (0) is a
        *persistent* lock - the sliders take direct control and it will
        NOT silently resume automatic like the key-nudges do. Flip it
        back to Auto (1) to hand control back to the automatic loop. The
        Zoom slider specifically only sticks while no leaf is currently
        being tracked - once a leaf is found, auto-zoom takes over zoom/
        pan for as long as it's tracked.

Press 'm' to toggle a small debug inset showing the raw detection mask -
useful if the bounding box ever seems to disappear, since it shows
exactly what the detector is (or isn't) seeing.

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
        Zoom           - digital zoom, 100 = 1.0x ... 300 = 3.0x (only
                          sticks while no leaf is being tracked - auto-
                          zoom takes over once one is found)
        Auto Mode      - 1 = automatic exposure (default), 0 = manual
                          (locks control to the Exposure/Brightness
                          sliders until switched back to 1)
"""

import cv2
import numpy as np
import os
import time
import ctypes
from collections import deque
from dataclasses import dataclass
from datetime import datetime

# rembg is an OPTIONAL dependency used only for the one-shot capture-time
# crop refinement (see refine_bbox_with_rembg) - never in the per-frame
# live loop, since it's far too slow for that (see yolo_poc/ evaluation:
# ~0.7-1.5 FPS vs ~9-10 FPS for the contour detector). If it isn't
# installed, refinement is silently skipped and everything else in this
# file behaves exactly as before.
try:
    from rembg import remove as _rembg_remove, new_session as _rembg_new_session
    REMBG_AVAILABLE = True
except ImportError:
    _rembg_remove = None
    _rembg_new_session = None
    REMBG_AVAILABLE = False


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

    # --- Generic fallback / hard floor. The LEARNED adaptive target is
    # never allowed to relax below these on its own - but see
    # quality_floor_relax_frac below, which progressively lowers this
    # floor itself the longer a leaf sits present without a capture, so a
    # floor that's simply uncalibrated for a given camera/lens (rather
    # than a real "this frame is too blurry" case) can't block every
    # single capture forever. ---
    sharpness_threshold: float = 90.0
    vein_score_threshold: float = 12.0
    # By the time a leaf has been continuously present for
    # capture_timeout_sec without a capture, the hard floor above has
    # relaxed by this fraction (e.g. 0.2 = up to 20% lower). Ramps
    # linearly from 0 (leaf just appeared - full strictness, favors a
    # genuinely sharp capture if one is quickly achievable) up to this
    # fraction. Kept fairly small deliberately: this lens is fixed-focus
    # (see AdaptiveThresholds), so a frame that's actually out of focus
    # stays out of focus no matter how long you wait - relaxing the floor
    # too far (0.4, previously) let genuinely blurry frames pass the
    # AUTO/steady capture path instead of only ever reaching the visibly-
    # tagged timeout fallback below.
    quality_floor_relax_frac: float = 0.2

    # --- Stability / autocapture behaviour ---
    stability_frames_required: int = 8    # ~0.25-0.3s of "all checks pass"
    motion_threshold: float = 4.0         # fallback only, before adaptive learns the real noise floor
    capture_cooldown_sec: float = 3.0
    # guarantee: save best-seen shot if "perfect" never hits this long.
    # Longer than before (5.0) - this lens is fixed-focus, so finding the
    # one real sharp distance takes the user a moment; cutting the window
    # short pushed a capture out before a genuinely sharp frame had a
    # chance to happen.
    capture_timeout_sec: float = 8.0

    # --- Fixed exclusion zone - the bottom strip of the frame is where
    # THIS camera's own mount/tripod is physically visible in every single
    # frame, regardless of scene content. No amount of color/shape gate
    # tuning reliably rejects it every time (it's real hardware, not scene
    # clutter, and can pick up enough of a color cast under some lighting
    # to slip past the green/saturation/luma gates below) - but since it's
    # a FIXED region that can never contain the leaf, it's far more robust
    # to exclude it from detection entirely, before any color math runs at
    # all. Set to comfortably cover the visible mount hardware; 0.0
    # disables this (no region excluded). ---
    ignore_bottom_frac: float = 0.22

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
    # Current crop center as a fraction of the full frame (0-1). Owned by
    # the auto-zoom logic in main() while a leaf is being tracked; reset
    # to frame-center whenever the leaf is lost / nothing is being tracked.
    pan_x: float = 0.5
    pan_y: float = 0.5

    # --- Auto-zoom: narrow the field of view onto the leaf BEFORE relying
    # on it for anything else, so a wide/cluttered shot (background,
    # tripod, other objects) gets cropped away and only a clean,
    # leaf-dominated view ever reaches the detection/quality/capture logic
    # below - the goal being "frame it like a human would before deciding
    # anything about it looks like". ---
    # Target area-ratio band to zoom toward - matches quality_score's own
    # "optimal" framing band (min_area_ratio + 0.25..0.65 of the range),
    # so "well framed by auto-zoom" and "scores well" can't disagree.
    auto_zoom_target_low: float = 0.15
    auto_zoom_target_high: float = 0.35
    # How far off-center (as a fraction of frame width/height) the leaf
    # may be before auto-zoom considers it FULLY "aimed" (panning stops
    # adjusting once inside this).
    auto_zoom_center_tol: float = 0.10
    # A SEPARATE, LOOSER tolerance that only gates whether zoom is allowed
    # to run at all - NOT the same as auto_zoom_center_tol above, which
    # only controls when panning stops fine-adjusting. apply_digital_zoom
    # is a complete no-op while zoom_factor<=1.001 (see its own docstring),
    # so panning has ZERO visible effect on the frame until zoom has
    # already taken its first step. If zoom were gated behind the tight
    # `centered` check using the SAME tolerance as panning, a leaf that
    # starts out more than auto_zoom_center_tol off-center could never be
    # centered by panning (since panning does nothing yet) and zoom could
    # never start - a permanent deadlock. This looser tolerance lets zoom
    # begin as soon as the leaf is roughly aimed at, while still refusing
    # to zoom when it's close enough to a frame edge that zooming could
    # crop it out entirely - the actual risk this check exists to prevent.
    auto_zoom_safe_to_zoom_tol: float = 0.30
    auto_zoom_step: float = 0.04        # zoom_factor change per frame
    auto_pan_step: float = 0.06         # max pan-fraction shift per frame
    # If the leaf has been missing this long, give up and reset back to a
    # neutral wide view to search again, rather than staying zoomed in on
    # empty space.
    auto_zoom_lost_reset_sec: float = 2.0
    # If zoom has been pushing the SAME direction (in, or out) for this
    # long without area_ratio actually improving by at least
    # auto_zoom_stall_improve_eps, stop pushing further rather than
    # continuing indefinitely - guards against a runaway zoom-in that
    # never actually converges (e.g. panning lagging behind a tight crop,
    # so the visible leaf fraction never grows the way zooming-in expects
    # it to).
    auto_zoom_stall_timeout_sec: float = 2.5
    auto_zoom_stall_improve_eps: float = 0.015

    # --- Cluttered-background robustness ---
    # Candidate contours whose w/h ratio falls outside this range are
    # rejected outright before scoring - real leaves don't come as thin
    # slivers, but background clutter edges (mesh wires, table seams,
    # shadow bands) often do.
    leaf_aspect_min: float = 0.12
    leaf_aspect_max: float = 8.0
    # Candidate SCORING saturates area's contribution at this fraction of
    # the frame - a candidate at or beyond this size scores the same on
    # the "size" term as one exactly at this ratio, it gets no further
    # reward for being bigger still. Without this, area_ratio multiplies
    # the score directly and unboundedly (up to the hard area ceiling
    # below), so a large merged background+leaf blob will always
    # out-score a smaller, more precisely-green true leaf just by being
    # bigger - this is what let "most of the frame" win as "the leaf".
    # Set to match the app's OWN "well-framed" zone (quality_score's
    # optimal_low, ~min_area_ratio + 0.25*(max_area_ratio-min_area_ratio)
    # =~ 0.15) - a leaf that size or larger is already "big enough",
    # so it shouldn't need to keep growing (or fusing with background)
    # to out-score a same-size-or-bigger competitor; once both are
    # saturated, color/shape confidence alone decides, which reliably
    # favors a clean leaf silhouette over a diluted merged blob.
    leaf_size_score_saturation: float = 0.15
    # Switch hysteresis (see _pick_with_hysteresis): once a track exists,
    # a DIFFERENT candidate must score at least this fraction higher
    # before the detector switches to it - e.g. 0.2 means 20% better.
    # Without this, two similarly-scored blobs (or the same object's
    # contour splitting slightly differently frame to frame) makes the
    # box visibly hop between them.
    continuity_switch_margin: float = 0.2
    # Display-only smoothing of the drawn box (see LeafTracker) - purely
    # cosmetic, does not affect what gets measured/saved.
    bbox_smooth_alpha: float = 0.45
    bbox_hold_frames: int = 4

    # --- Flicker ("random light flashing") smoothing ---
    # Both the exposure controller and the gamma post-process react to a
    # brightness reading; feeding them the raw single-frame value lets
    # one noisy/clutter-confused frame yank exposure or gamma around,
    # which reads on screen as a brightness "flash". EMA-smoothing the
    # readings first fixes that without adding perceptible lag.
    brightness_ema_alpha: float = 0.25
    gamma_ema_alpha: float = 0.2
    # The brightness reading source flips between "whole frame" (no leaf)
    # and "leaf-only ROI" (leaf found) - a discontinuous jump right when a
    # leaf enters/leaves frame. For the first N frames after that flip,
    # use a slower EMA alpha so the exposure/gamma reaction to the jump
    # itself is smoothed out, not just ordinary per-frame noise.
    brightness_transition_alpha: float = 0.08
    brightness_transition_hold_frames: int = 15

    # --- Cluttered-background detection tuning ---
    # A candidate only gets the temporal-continuity score boost once the
    # tracker has held the SAME object for this many consecutive frames -
    # otherwise a bad lock in the first frame or two would keep
    # reinforcing itself every frame after (continuity favors "whatever we
    # were already tracking", which is only trustworthy once it's proven
    # itself over a few frames).
    continuity_min_confirmed_frames: int = 5
    # Every N frames, force one "cold" comparison with continuity disabled
    # so a better candidate elsewhere in frame (e.g. the real leaf finally
    # entering view after the tracker locked onto clutter) can win instead
    # of being permanently out-scored by the continuity bonus.
    cold_recheck_interval_frames: int = 90

    # --- Capture refinement/rejection ---
    # Once the stability window first completes, keep evaluating for this
    # many extra frames and save the best-scoring one of them, instead of
    # whichever frame happened to complete the streak first.
    capture_refine_frames: int = 3
    # If True, the timeout guarantee (capture_timeout_sec) never saves a frame below
    # low_confidence_score - it keeps waiting (and restarts its own timer)
    # instead. If False (default), it saves the best-seen frame anyway so
    # a photo is guaranteed, flagged "timeout"/low-confidence.
    strict_reject_blurry: bool = False

    # --- Capture-time crop refinement (rembg) ---
    # rembg is class-agnostic salient-object segmentation - it doesn't
    # know what a "leaf" is, just "the most visually prominent single
    # foreground region", which our offline evaluation (yolo_poc/) showed
    # handles cluttered backgrounds noticeably better than the contour
    # method: 100% vs 89% detection rate across real test captures, and
    # in several cases where the two disagreed, rembg was the one that
    # got it right (the contour method had boxed a shadow or nearly the
    # whole frame). It's far too slow to run every frame (~0.7-1.5 FPS),
    # so it only runs ONCE, right before a photo is actually saved, to
    # refine/validate the final crop - never in the live preview loop.
    use_rembg_refinement: bool = True
    rembg_model: str = "u2netp"   # lightweight variant - ~2x faster than u2net, same detection rate in testing
    rembg_mask_thresh: int = 127
    # rembg's box is only trusted over the contour method's if the region
    # it picked is itself green-dominant (same gate _score_candidates
    # uses) - guards against rembg confidently segmenting something that
    # isn't the leaf at all (a hand, a reflection, a shadow), since it has
    # no notion of "leaf" to begin with.
    rembg_color_margin_floor: float = 1.3

    # --- Live-preview tracker supervision (rembg) ---
    # rembg's capture-time refinement above only fixes the FINAL saved
    # crop - it says nothing about the box shown on screen while framing,
    # which still comes entirely from the fast contour method and can
    # still occasionally lock onto the wrong object (and, via switch
    # hysteresis, keep defending that lock rather than self-correcting).
    # Every rembg_live_recheck_interval_sec, consult rembg ONCE on the
    # live frame and force-reseed the tracker if it disagrees with what's
    # currently tracked (or if nothing is currently tracked at all).
    #
    # DEFAULTED OFF: live-tested and made detection noticeably worse - the
    # tracker gets force-reset every 2s, but the next few frames of
    # ordinary contour tracking (which runs WITHOUT hysteresis right after
    # a reset, see detect_leaf/apply_hysteresis) can drag the smoothed box
    # straight back toward whatever the contour method was already biased
    # toward before the next correction lands, fighting itself rather than
    # converging. The capture-time refinement above (use_rembg_refinement)
    # was validated as a real improvement and is unaffected by this flag -
    # only re-enable this one if you want to experiment further.
    rembg_live_supervision: bool = False
    rembg_live_recheck_interval_sec: float = 2.0

    # --- Debug ---
    show_debug_mask: bool = False


# --------------------------------------------------------------------------- #
# Leaf detection - works on any background
# --------------------------------------------------------------------------- #

def estimate_background_color(frame, margin=50):
    """Median color sampled from a ring of border patches. This is a
    last-resort estimate (see detect_leaf) precisely because it assumes a
    roughly uniform background - true of a plain mat/table, false of a
    genuinely cluttered scene. Sampling all four corners/edges (not just
    the top) at least keeps it from being fooled by a background that's
    uniform along one side but not another."""
    h, w = frame.shape[:2]
    patches = [
        frame[0:margin, 0:margin],
        frame[0:margin, w - margin:w],
        frame[h - margin:h, 0:margin],
        frame[h - margin:h, w - margin:w],
        frame[0:margin, w // 2 - margin:w // 2 + margin],
        frame[h - margin:h, w // 2 - margin:w // 2 + margin],
        frame[h // 2 - margin:h // 2 + margin, 0:margin],
        frame[h // 2 - margin:h // 2 + margin, w - margin:w],
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
    acceptable trade because _score_candidates (below) then screens
    candidates on shape, solidity, AND color plausibility rather than
    blindly taking the largest blob."""
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_u8 = np.clip(exg, 0, 255).astype(np.uint8)
    exg_blur = cv2.GaussianBlur(exg_u8, (5, 5), 0)
    otsu_val, _ = cv2.threshold(exg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inclusive_thresh = min(otsu_val, 12.0)
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


def background_diff_mask(frame, bg_color):
    diff = np.linalg.norm(frame.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
    _, mask = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _bbox_iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes, 0.0 if disjoint
    or either box is missing/degenerate."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _score_candidates(frame, mask, frame_area, max_ratio=0.75, min_ratio=0.003, min_solidity=0.0,
                       aspect_min=None, aspect_max=None, size_saturation_ratio=0.35, top_k=3,
                       debug=False, debug_label=""):
    """Scores every plausible contour in `mask` on its own merits (size,
    color, shape only - no notion of tracking/continuity here, see
    detect_leaf's _pick_with_hysteresis for that) and returns the top_k
    best as a list of dicts (sorted best-first): {mask, bbox, area_ratio,
    score}. Returning more than one candidate - instead of just a single
    winner - is what lets detect_leaf() compare candidates ACROSS detector
    tiers (ExG vs HSV) rather than blindly accepting whichever tier
    happens to run first; see detect_leaf's docstring.

    Guards applied to every candidate before it's even scored:
      - area_ratio must fall within [min_ratio, max_ratio].
      - aspect_min/aspect_max reject thin slivers (mesh wire, table seam,
        shadow band) - real leaves don't come that elongated.
      - a candidate touching 3+ of the 4 frame edges is rejected outright,
        regardless of area_ratio - background, or several objects merged
        into one blob, not an isolated leaf.
      - min_solidity rejects rough/jagged/branching blobs.
      - color margin (green minus red/blue) must be green-dominant, with
        BOTH an absolute floor and a floor relative to the candidate's own
        brightness. A fixed absolute cutoff alone under-rejects noise in
        bright scenes and over-rejects real leaves in dim ones, since raw
        color differences compress toward zero as brightness drops.
      - a per-pixel green_fraction check: at least 55% of the candidate's
        OWN pixels must individually clear a green-dominance floor, not
        just the region's mean. This is what actually rejects a MERGED
        blob (a real leaf fused by morphological close with an adjacent
        swath of weakly-green background, e.g. a wood table) - the mean
        color-margin check above can still pass a merged blob if the real
        leaf pixels alone drag its average past the floor, even though
        most of the blob's area isn't leaf. Confirmed via debug output
        (see detect_leaf's debug=True path) that this was letting a large
        merged blob outscore the correctly-shaped, correctly-positioned
        real leaf candidate purely on size.
      - an explicit minimum LUMA rejects genuinely dark/black objects
        directly by how dark they are, rather than relying on color-margin
        math alone (noisy in dark regions).
      - an explicit minimum SATURATION additionally rejects achromatic
        objects even when auto_gamma_correct's brightness scaling has
        pushed their luma up (scaling all channels equally changes luma
        but not hue/saturation, so saturation stays a reliable "is this
        actually colored" signal even during an under/over-exposure
        hunting phase where luma alone can be misleading).

    Scoring, for whatever survives the gates:
      - base = size_factor x (1 + color_margin factor) - size_factor is
        area_ratio SATURATING at size_saturation_ratio (see
        Config.leaf_size_score_saturation), not the raw ratio. A large
        merged background+leaf blob and a well-framed true leaf that are
        both at or beyond that saturation point score identically on
        size - the blob no longer wins purely by being bigger.
      - x circularity factor (4*pi*area/perimeter^2, capped at 1.0) - real
        leaves are reasonably compact; jagged/branching clutter (mesh,
        cast shadows, wires) scores lower here even if it slipped past
        the solidity gate."""
    # A smaller, tighter close (and a separate, smaller open) than before -
    # the previous 9x9/2-iteration close was aggressive enough to bridge
    # small gaps between the true leaf and adjacent background pixels that
    # only marginally passed the color threshold, FUSING them into one
    # connected contour before scoring ever saw them as separate objects.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print(f"    [{debug_label}] 0 raw contours in mask")
        return []

    # Computed once per call (not per-candidate) - saturation is the S
    # channel of HSV, sampled per-candidate below.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_h, mask_w = mask.shape[:2]
    edge_margin_x = max(int(mask_w * 0.02), 2)
    edge_margin_y = max(int(mask_h * 0.02), 2)

    # Per-gate rejection counters - printed as a summary when debug=True
    # (see Config.show_debug_mask / the 'm' key) so it's directly visible
    # HOW MANY contours existed, and exactly which gate rejected each one
    # that didn't survive, rather than only ever seeing the final winner.
    rejected = {"area_ratio": 0, "aspect": 0, "border_touch": 0,
                "solidity": 0, "luma": 0, "color_margin": 0, "mixed_region": 0, "saturation": 0}

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        area_ratio = area / frame_area
        if area_ratio < min_ratio or area_ratio > max_ratio:
            rejected["area_ratio"] += 1
            continue

        x, y, w, h = cv2.boundingRect(c)
        if aspect_min is not None and aspect_max is not None and h > 0:
            aspect = w / h
            if aspect < aspect_min or aspect > aspect_max:
                rejected["aspect"] += 1
                continue  # too sliver-like to plausibly be a leaf

        # A candidate touching 3+ of the 4 frame edges spans nearly the
        # whole frame - background, or several objects merged into one
        # blob, not an isolated leaf. area_ratio alone can't tell "large
        # but isolated" from "spans edge to edge"; this geometric check
        # can, and it's exactly what a "very big box covering the leaf and
        # other objects" looks like.
        touches = ((x <= edge_margin_x) + (x + w >= mask_w - edge_margin_x)
                   + (y <= edge_margin_y) + (y + h >= mask_h - edge_margin_y))
        if touches >= 3:
            rejected["border_touch"] += 1
            continue

        if min_solidity > 0:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < min_solidity:
                rejected["solidity"] += 1
                continue

        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(candidate_mask, [c], -1, 255, thickness=cv2.FILLED)

        means = _mean_bgr_in_mask(frame, candidate_mask)
        if means is None:
            continue
        cb, cg, cr = means
        color_margin = min(cg - cr, cg - cb)
        mean_luma = (cb + cg + cr) / 3.0
        if mean_luma < 35.0:
            # Explicit brightness floor - a genuinely dark/black object (a
            # cable, a clip, a shadow, a tripod/camera part) is rejected
            # directly by HOW DARK IT IS, rather than relying solely on
            # the color-margin math below. Dark, low-signal image regions
            # are noisy (more so once amplified by sensor gain), so a
            # near-black object's per-pixel color can average out to a
            # small but non-zero green margin by pure chance even though
            # it obviously isn't a leaf - this is what let dark objects
            # get accepted as "the leaf" with the color-margin gate alone.
            rejected["luma"] += 1
            continue
        relative_margin = color_margin / max(mean_luma, 1.0)
        if color_margin <= 10.0 or relative_margin <= 0.04:
            # Raised from 4.0 based on per-candidate debug metrics (mean_bgr/
            # color_margin/green_fraction, printed for every survivor) captured
            # live: a large, weakly-green background/wall region consistently
            # measured color_margin 6-9 across dozens of frames and won purely
            # on size (10-15x the real leaf's area), while the actual leaf -
            # in the same frames - consistently measured color_margin >= 10.2.
            # 4.0 was loose enough to accept that background blob as "the
            # leaf" just for being the biggest candidate around; 10.0 sits
            # in the clean gap between the two populations seen in that data.
            rejected["color_margin"] += 1
            continue  # not actually green-dominant - never let this win on size alone

        # PER-PIXEL uniformity check - color_margin/relative_margin above
        # only look at the MEAN color of the whole candidate, which a
        # MERGED region (a real leaf fused by morphological close with an
        # adjacent swath of weakly-green background - e.g. a wood table
        # under lighting where ExG's own deliberately-inclusive threshold
        # reads it as faintly green) can still pass even though most of
        # its area isn't leaf at all: the real leaf pixels alone can drag
        # the average past every mean-based floor above. Confirmed via
        # debug output: a ~20%-of-frame merged blob outscored the actual,
        # correctly-shaped, correctly-positioned leaf candidate sitting
        # right next to it purely because size_factor rewards its bulk -
        # tightening the MEAN gates further can't fix that, since the
        # problem isn't "the average is slightly off", it's "the average
        # is computed over a mostly-non-leaf region". Requiring a HIGH
        # FRACTION of the candidate's own pixels to individually clear a
        # green-dominance floor (not just the region's mean) directly
        # rejects a mostly-background blob regardless of how its average
        # happens to work out, while a genuine, solid leaf easily passes
        # since nearly all of its pixels are actually green.
        region_pixels = frame[candidate_mask > 0].astype(np.float32)
        region_margin = np.minimum(region_pixels[:, 1] - region_pixels[:, 2],
                                    region_pixels[:, 1] - region_pixels[:, 0])
        green_fraction = float(np.mean(region_margin > 3.0))
        if green_fraction < 0.55:
            rejected["mixed_region"] += 1
            continue

        sat_values = hsv_frame[:, :, 1][candidate_mask > 0]
        if sat_values.size == 0 or float(np.mean(sat_values)) < 25.0:
            # Saturation is a MORE ROBUST signal than the luma floor above
            # against one specific failure mode: auto_gamma_correct applies
            # a uniform multiplicative brightness scale to the whole frame
            # BEFORE detection ever runs - multiplying all three (B,G,R)
            # channels by the same factor changes luma but leaves hue/
            # saturation exactly unchanged (mathematically: scaling all
            # channels equally preserves their ratios). So during an
            # underexposed stretch, gamma correction's brightening can push
            # a genuinely black/achromatic object's LUMA up past the floor
            # above, while its saturation - correctly reflecting that it
            # has no real color - stays low regardless. This is what let a
            # "black object" slip through the luma floor during exactly
            # the early under/over-exposure hunting phase where it matters
            # most.
            rejected["saturation"] += 1
            continue

        perimeter = cv2.arcLength(c, True)
        circularity = min((4.0 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0
        shape_factor = 0.5 + 0.5 * circularity

        size_factor = min(area_ratio / size_saturation_ratio, 1.0) if size_saturation_ratio > 0 else area_ratio
        score = size_factor * (1.0 + min(color_margin, 60.0) / 60.0) * shape_factor
        candidates.append({"mask": candidate_mask, "bbox": (x, y, w, h),
                            "area_ratio": area_ratio, "score": score,
                            "color_margin": color_margin, "green_fraction": green_fraction,
                            "mean_sat": float(np.mean(sat_values)), "mean_bgr": (cb, cg, cr)})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if debug:
        survived = len(candidates)
        top = candidates[0]["bbox"] if candidates else None
        top_score = candidates[0]["score"] if candidates else None
        print(f"    [{debug_label}] {len(contours)} raw contours -> {survived} survived all gates "
              f"(rejected: {rejected}) -> top candidate bbox={top} score={top_score}")
        # Per-candidate metrics for every survivor (not just the winner) -
        # this is what actually shows WHY a candidate won: whether it's
        # genuinely strong on color/purity or just large (size_factor),
        # and lets a false-positive winner's color_margin/green_fraction be
        # compared directly against the true leaf candidate sitting right
        # next to it in the same list.
        for i, c in enumerate(candidates):
            print(f"      [{debug_label}] #{i} bbox={c['bbox']} area_ratio={c['area_ratio']:.4f} "
                  f"score={c['score']:.3f} color_margin={c['color_margin']:.1f} "
                  f"green_fraction={c['green_fraction']:.2f} mean_sat={c['mean_sat']:.1f} "
                  f"mean_bgr=({c['mean_bgr'][0]:.0f},{c['mean_bgr'][1]:.0f},{c['mean_bgr'][2]:.0f})")
    return candidates[:top_k]


def _pick_with_hysteresis(candidates, prev_bbox, switch_margin, min_iou=0.15):
    """Chooses which of `candidates` (already-scored, non-empty) to use
    this frame. If prev_bbox is given and some candidate substantially
    overlaps it (the object we were already tracking), that candidate
    wins UNLESS a DIFFERENT candidate scores at least switch_margin
    higher - this is what stops the box hopping between two similarly-
    scored blobs, or between two slightly different contour splits of the
    same object, on every other frame. A genuinely better candidate still
    wins; if nothing overlaps prev_bbox at all (the tracked object is
    really gone, or hysteresis is disabled by passing prev_bbox=None),
    the plain best-scoring candidate wins with no resistance."""
    best = max(candidates, key=lambda c: c["score"])
    if prev_bbox is None:
        return best

    tracked = max(candidates, key=lambda c: _bbox_iou(c["bbox"], prev_bbox))
    if _bbox_iou(tracked["bbox"], prev_bbox) < min_iou:
        return best  # nothing here is really "the same object" any more

    if best is tracked or best["score"] <= tracked["score"] * (1.0 + switch_margin):
        return tracked  # not a big enough improvement - keep what we had
    return best  # genuinely better - allow the switch


def detect_leaf(frame, cfg: Config, prev_bbox=None, apply_hysteresis=True, debug=False):
    """Detection order:

    0. Config.ignore_bottom_frac blanks out a fixed strip at the bottom of
       every mask before anything else runs - this camera's own mount/
       tripod is physically visible there in every frame, and no color/
       shape gate reliably rejects real hardware every single time, so it
       is excluded at the source instead.

    1. Excess-green (ExG) segmentation - the primary detector. Robust to
       background AND to skin tone, since it reasons about green-channel
       dominance rather than a fixed hue window.
    2. HSV true-green hue mask - secondary check, catches cases ExG
       misses (e.g. very low-saturation lighting).

    Candidates from tiers 1 and 2 are pooled and compared TOGETHER - the
    winner is whichever candidate scores highest across BOTH tiers, not
    just whichever tier happened to run first. This matters in clutter:
    previously, if ExG's single best guess was a background object, the
    real leaf was never even considered even if it was a strong candidate
    (in ExG as a runner-up, or in HSV).

    3. Background-difference - true last resort, only tried when 1 and 2
       together produce nothing at all. It assumes a roughly uniform
       background (sampled from the frame border), which is usually false
       in genuinely cluttered scenes, so it's deliberately the weakest
       option rather than a peer of 1 and 2.

    prev_bbox/apply_hysteresis implement switch hysteresis (see
    _pick_with_hysteresis): once a track exists, a different candidate
    must score meaningfully higher before the detector switches to it,
    which is what stops the box hopping between two similarly-scored
    blobs frame to frame. The caller (main loop) is expected to pass
    apply_hysteresis=False until a track is "confirmed" (held for several
    consecutive frames) and periodically force it False again for a
    "cold" recheck, so an early bad lock can't permanently reinforce
    itself - hysteresis never overrides the color/shape gates, only which
    already-valid candidate wins.

    Returns (mask_used_for_detection, bbox_or_None, area_ratio, debug_info).
    mask_used_for_detection is always returned (even on failure) so the
    caller can show it in the debug inset. debug_info is None when no
    leaf was found, otherwise a dict with "score" (the winning
    candidate's score), "raw_mask_fraction" (how much of the frame the
    RAW per-pixel mask covers, before contour selection - see fix #5,
    this is the number that reveals a mask that's over-including
    background even when the SELECTED contour looks reasonable) and
    "tier" (which detector stage won)."""
    frame_area = frame.shape[0] * frame.shape[1]
    hyst_bbox = prev_bbox if apply_hysteresis else None

    # max_ratio caps how much of the frame a single candidate is allowed
    # to cover before it's rejected outright - this is the primary guard
    # against background merging with the leaf into one big blob and
    # getting accepted as "the leaf". Set just above cfg.max_area_ratio
    # (the "too close, move away" guidance threshold) so a legitimately
    # large/close leaf still gets detected, but nowhere near "most of the
    # frame" can ever pass. Tightened from +0.10/0.65 - that gave enough
    # headroom for a background-merged blob to still sneak under the
    # ceiling.
    max_candidate_ratio = min(cfg.max_area_ratio + 0.05, 0.60)

    # Blank out the fixed exclusion zone (see Config.ignore_bottom_frac)
    # in EVERY mask before any contour is ever extracted from it - the
    # camera's own mount/tripod, physically visible in the same region of
    # every frame, can never contain the leaf, so it's excluded at the
    # source rather than relying on color/shape gates to reject it anew
    # every single frame.
    frame_h, frame_w = frame.shape[:2]
    roi_mask = np.full((frame_h, frame_w), 255, dtype=np.uint8)
    cutoff = frame_h
    if cfg.ignore_bottom_frac > 0:
        cutoff = int(frame_h * (1.0 - cfg.ignore_bottom_frac))
        roi_mask[cutoff:, :] = 0

    raw_exg = excess_green_mask(frame)
    exg_mask = cv2.bitwise_and(raw_exg, roi_mask)
    raw_hsv_for_exclusion = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    raw_hue = cv2.inRange(raw_hsv_for_exclusion, lower, upper)
    hue_mask = cv2.bitwise_and(raw_hue, roi_mask)

    if debug:
        # Direct, falsifiable proof the exclusion zone is removing real
        # foreground pixels (not just a visual overlay) - counts pixels
        # that WOULD have been foreground in the excluded strip, per mask,
        # before roi_mask zeroed them out.
        if cfg.ignore_bottom_frac > 0:
            exg_excluded = int(np.count_nonzero(raw_exg[cutoff:, :]))
            hue_excluded = int(np.count_nonzero(raw_hue[cutoff:, :]))
        else:
            exg_excluded = hue_excluded = 0
        print(f"  [exclusion] cutoff_y={cutoff}/{frame_h} ({cfg.ignore_bottom_frac*100:.0f}% of frame) "
              f"exg_pixels_removed={exg_excluded}  hue_pixels_removed={hue_excluded}")

    exg_candidates = _score_candidates(frame, exg_mask, frame_area, max_ratio=max_candidate_ratio,
                                        min_ratio=0.003, min_solidity=0.28,
                                        aspect_min=cfg.leaf_aspect_min, aspect_max=cfg.leaf_aspect_max,
                                        size_saturation_ratio=cfg.leaf_size_score_saturation, top_k=3,
                                        debug=debug, debug_label="exg")

    hue_candidates = _score_candidates(frame, hue_mask, frame_area, max_ratio=max_candidate_ratio,
                                        min_ratio=0.003, min_solidity=0.28,
                                        aspect_min=cfg.leaf_aspect_min, aspect_max=cfg.leaf_aspect_max,
                                        size_saturation_ratio=cfg.leaf_size_score_saturation, top_k=3,
                                        debug=debug, debug_label="hue")

    pooled = exg_candidates + hue_candidates
    if pooled:
        winner = _pick_with_hysteresis(pooled, hyst_bbox, cfg.continuity_switch_margin)
        raw_frac = float(np.count_nonzero(cv2.bitwise_or(exg_mask, hue_mask))) / frame_area
        debug_info = {"score": winner["score"], "raw_mask_fraction": raw_frac, "tier": "exg+hsv"}
        if debug:
            wx, wy, ww, wh = winner["bbox"]
            print(f"  [detect_leaf] SELECTED tier=exg+hsv bbox=({wx},{wy},{ww},{wh}) "
                  f"bottom_edge_y={wy + wh} vs cutoff_y={cutoff} "
                  f"score={winner['score']:.2f} area_ratio={winner['area_ratio']:.4f}")
        return winner["mask"], winner["bbox"], winner["area_ratio"], debug_info

    bg_color = estimate_background_color(frame)
    raw_bg = background_diff_mask(frame, bg_color)
    bg_mask = cv2.bitwise_and(raw_bg, roi_mask)
    if debug:
        bg_excluded = int(np.count_nonzero(raw_bg[cutoff:, :])) if cfg.ignore_bottom_frac > 0 else 0
        print(f"  [exclusion] bg_diff tier: bg_pixels_removed={bg_excluded}")
    bg_candidates = _score_candidates(frame, bg_mask, frame_area, max_ratio=0.40,
                                       min_ratio=0.003, min_solidity=0.30,
                                       aspect_min=cfg.leaf_aspect_min, aspect_max=cfg.leaf_aspect_max,
                                       size_saturation_ratio=cfg.leaf_size_score_saturation, top_k=3,
                                       debug=debug, debug_label="bg_diff")
    if bg_candidates:
        winner = _pick_with_hysteresis(bg_candidates, hyst_bbox, cfg.continuity_switch_margin)
        raw_frac = float(np.count_nonzero(bg_mask)) / frame_area
        debug_info = {"score": winner["score"], "raw_mask_fraction": raw_frac, "tier": "bg_diff"}
        if debug:
            wx, wy, ww, wh = winner["bbox"]
            print(f"  [detect_leaf] SELECTED tier=bg_diff bbox=({wx},{wy},{ww},{wh}) "
                  f"bottom_edge_y={wy + wh} vs cutoff_y={cutoff} "
                  f"score={winner['score']:.2f} area_ratio={winner['area_ratio']:.4f}")
        return winner["mask"], winner["bbox"], winner["area_ratio"], debug_info

    if debug:
        print("  [detect_leaf] NO candidate survived any tier this frame")

    # Nothing passed validation - still hand back the combined raw masks
    # so the debug inset ('m' key) can show *why* nothing matched, instead
    # of a blank screen.
    combined = cv2.bitwise_or(exg_mask, cv2.bitwise_or(hue_mask, bg_mask))
    return combined, None, 0.0, None


# --------------------------------------------------------------------------- #
# Temporal bbox tracking - smooths the DRAWN box and briefly holds it
# through a one- or two-frame detection dropout (common in clutter, where a
# candidate contour can flicker in/out of the winning score by a hair from
# one frame to the next). Display-only: guidance, quality scoring, and what
# gets saved all continue to use the raw, un-smoothed per-frame detection -
# this only stops the on-screen box from visibly jittering/vanishing.
# --------------------------------------------------------------------------- #

class LeafTracker:
    def __init__(self, smooth_alpha: float, hold_frames: int):
        self.smooth_alpha = smooth_alpha
        self.hold_frames = hold_frames
        self.smoothed_bbox = None
        self.frames_since_seen = 0
        # Consecutive frames the SAME track has been held (survives brief
        # hold-frame flicker, resets on a real loss). Used by the main
        # loop to decide when continuity scoring is trustworthy enough to
        # enable (see Config.continuity_min_confirmed_frames) and when the
        # smoothed box is stable enough to drive measurement/capture, not
        # just the display (see Config.continuity_min_confirmed_frames
        # usage in main()).
        self.confirmed_frames = 0

    def update(self, bbox):
        """Feed this frame's raw detection (or None). Returns the box to
        draw: exponentially smoothed while the leaf is seen, held steady
        for up to hold_frames after it briefly drops out, then cleared."""
        if bbox is not None:
            if self.smoothed_bbox is None:
                self.smoothed_bbox = tuple(float(v) for v in bbox)
            else:
                a = self.smooth_alpha
                self.smoothed_bbox = tuple(
                    a * nv + (1 - a) * ov for nv, ov in zip(bbox, self.smoothed_bbox))
            self.frames_since_seen = 0
            self.confirmed_frames += 1
        else:
            self.frames_since_seen += 1
            if self.frames_since_seen > self.hold_frames:
                self.smoothed_bbox = None
                self.confirmed_frames = 0

        if self.smoothed_bbox is None:
            return None
        return tuple(int(round(v)) for v in self.smoothed_bbox)


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

def apply_digital_zoom(frame, zoom_factor, center_x=0.5, center_y=0.5):
    """center_x/center_y (fractions of the full frame, 0-1) let the crop
    follow a subject instead of always being centered on the frame - this
    is what lets auto-zoom pan TOWARD the leaf as it zooms in, rather than
    zooming in place and potentially cropping an off-center leaf out of
    frame entirely."""
    if zoom_factor <= 1.001:
        return frame
    h, w = frame.shape[:2]
    crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)
    cx, cy = int(w * center_x), int(h * center_y)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Keyboard diagnostics - lets us confirm whether cv2.waitKey is actually
# receiving keypresses at all, and which window is focused when it does.
# waitKey only delivers a key to THIS process if an OpenCV HighGUI window
# (not the console, not another app) has OS focus, so "q/s not working" is
# very often just a focus problem rather than a bug in the key-handling
# code below.
# --------------------------------------------------------------------------- #

def get_foreground_window_title():
    """Best-effort title of whichever window currently has OS input focus.
    Windows-only (uses ctypes/user32); returns None everywhere else or on
    any failure, so callers should treat a None as "unknown", not "no
    window focused"."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return None


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

    Reliability additions:
    3. Not every property is actually settable on every UVC driver -
       cap.set() silently no-ops on many webcams instead of raising, which
       previously meant the loop could print "raising"/"stable" forever
       while nothing physically changed. set_baseline() now probes each of
       EXPOSURE/GAIN/BRIGHTNESS with a small test change and only budgets
       adjustments to properties that actually moved.
    4. Step size used to be a fixed absolute delta (e.g. always "+4" on
       GAIN), which is meaningless without knowing that driver's real
       range - too slow to converge on some cameras, too twitchy on
       others. Steps are now a fraction of each property's own configured
       drift band (*_step_frac), so convergence speed is consistent
       regardless of the underlying units.
    5. A small leaky-integral term is blended with the proportional error
       so a persistent (not just momentary) under/over-exposure pushes
       harder over time, damping the hunting a pure-proportional loop is
       prone to - especially now that the input itself is EMA-smoothed
       upstream (see brightness_ema_alpha), which adds a bit of lag.
    6. The startup baseline used to be permanent for the whole session: if
       lighting changed drastically (room -> sunlit window) the drift
       ceiling around that one baseline could permanently block reaching
       the new optimum. If a property stays pinned at its drift-band edge
       for a sustained stretch, that property's baseline is re-centered
       toward the pinned side, freeing up headroom to keep converging.

    IMPORTANT: pass in the brightness measured over the LEAF region when
    one is visible, not the whole frame - otherwise a bright/dark
    background can pull exposure the wrong way for the leaf itself."""

    TARGET_LOW = 110.0
    TARGET_HIGH = 165.0
    TARGET_MID = (TARGET_LOW + TARGET_HIGH) / 2
    RESUME_AFTER_IDLE_SEC = 20.0
    PINNED_CHECKS_BEFORE_REBASELINE = 5  # ~5 check-intervals stuck at the drift edge

    def __init__(self, check_interval_frames: int = 6, max_step: float = 1.0,
                 exposure_max_drift: float = 10.0, gain_max_drift: float = 150.0,
                 brightness_max_drift: float = 120.0,
                 exposure_step_frac: float = 0.22, gain_step_frac: float = 0.22,
                 brightness_step_frac: float = 0.22, integral_gain: float = 0.15):
        self.check_interval = check_interval_frames
        self.max_step = max_step
        self.integral_gain = integral_gain
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
        self.exposure_step_frac = exposure_step_frac
        self.gain_step_frac = gain_step_frac
        self.brightness_step_frac = brightness_step_frac
        self.exposure_supported = True
        self.gain_supported = True
        self.brightness_supported = True
        self.hardware_control_available = True
        self._black_frame_streak = 0
        self._white_frame_streak = 0
        self._error_integral = 0.0
        self._pinned_checks = {"exposure": 0, "gain": 0, "brightness": 0}
        self._auto_exposure_manual_value = None  # whichever convention (0.25 or 1) actually worked
        self._last_manual_reassert = 0.0
        self._ae_readback_reliable = True  # set for real in set_baseline()
        self._last_observed_ae = None

    def _probe_support(self, cap, prop, test_delta, epsilon):
        """Small deliberate change + readback to find out whether this
        driver actually honors writes to `prop`, instead of assuming it
        does (cap.set() silently no-ops on many drivers). Restores the
        original value before returning either way."""
        before = cap.get(prop)
        if before == -1:
            return False, before
        cap.set(prop, before + test_delta)
        after = cap.get(prop)
        cap.set(prop, before)
        supported = abs(after - before) > epsilon
        return supported, before

    def _verify_property_affects_frame(self, cap, prop, test_delta, flush_frames=5):
        """A readback match (_probe_support) isn't proof the hardware
        actually changed - some UVC drivers just echo back whatever you
        write without the sensor doing anything, which would otherwise
        look "supported" forever while never actually affecting the
        picture. This applies a real, sizeable change and checks whether
        freshly-grabbed FRAMES actually got measurably brighter/darker,
        which is the only way to really know. flush_frames gives the
        driver's internal buffer time to catch up to a new setting before
        each measurement."""
        def _mean_after_flush():
            ok, fr = False, None
            for _ in range(flush_frames):
                ok, fr = cap.read()
            if not ok or fr is None:
                return None
            return float(np.mean(fr))

        before_val = cap.get(prop)
        before_mean = _mean_after_flush()
        if before_mean is None:
            return False

        cap.set(prop, before_val + test_delta)
        after_mean = _mean_after_flush()

        cap.set(prop, before_val)
        _mean_after_flush()  # let it settle back before handing control back

        if after_mean is None:
            return False
        return abs(after_mean - before_mean) > 3.0

    def set_baseline(self, cap):
        """Call once after the camera has warmed up. Forces the camera out
        of its own auto-exposure mode, probes which properties this driver
        actually honors - both by readback AND by checking real frames
        actually change - and records the current exposure/gain/brightness
        as the safe anchor all later adjustments are bounded around."""
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow convention: 0.25 = manual
        self._auto_exposure_manual_value = 0.25
        if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) not in (0.25,):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 convention: 1 = manual
            self._auto_exposure_manual_value = 1
        ae_readback = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"  AUTO_EXPOSURE after manual-mode request: {ae_readback}")
        # Some drivers (seen on this project: DirectShow reporting -1
        # regardless of what's written) never confirm ANY value via
        # readback at all - readback isn't just "different", it's simply
        # not meaningful for this property on this driver. Detecting that
        # UP FRONT (rather than treating every future mismatch as "the
        # firmware reverted to auto") is what stops the periodic recheck
        # below from printing a false "reverted to auto mode" warning
        # every single check forever when nothing has actually changed -
        # that misleading noise was previously indistinguishable from a
        # genuine reversion event.
        self._ae_readback_reliable = abs(ae_readback - self._auto_exposure_manual_value) < 0.01
        self._last_observed_ae = ae_readback
        if not self._ae_readback_reliable:
            print("  NOTE: AUTO_EXPOSURE readback doesn't confirm manual mode on this driver "
                  "(common/harmless on some DirectShow webcams) - EXPOSURE/BRIGHTNESS writes "
                  "are what actually matter and are verified separately below; the periodic "
                  "re-assertion will keep writing this property defensively but won't warn "
                  "about it since this driver's readback isn't a reliable signal either way.")

        self.exposure_supported, self.baseline_exposure = self._probe_support(
            cap, cv2.CAP_PROP_EXPOSURE, 1.0, 0.05)
        self.gain_supported, self.baseline_gain = self._probe_support(
            cap, cv2.CAP_PROP_GAIN, 20.0, 1.0)
        self.brightness_supported, self.baseline_brightness = self._probe_support(
            cap, cv2.CAP_PROP_BRIGHTNESS, 20.0, 1.0)

        readback_str = (f"readback: exposure={'OK' if self.exposure_supported else 'no'} "
                         f"gain={'OK' if self.gain_supported else 'no'} "
                         f"brightness={'OK' if self.brightness_supported else 'no'}")
        print(f"  baseline exposure={self.baseline_exposure:.2f}  gain={self.baseline_gain:.2f}  "
              f"brightness={self.baseline_brightness:.2f}  ({readback_str})")

        # Readback alone can lie (echo-only drivers) - confirm each
        # property that passed readback ALSO visibly changes real frames.
        if self.exposure_supported:
            self.exposure_supported = self._verify_property_affects_frame(
                cap, cv2.CAP_PROP_EXPOSURE, 3.0)
        if self.gain_supported:
            self.gain_supported = self._verify_property_affects_frame(
                cap, cv2.CAP_PROP_GAIN, 60.0)
        if self.brightness_supported:
            self.brightness_supported = self._verify_property_affects_frame(
                cap, cv2.CAP_PROP_BRIGHTNESS, 60.0)
        self.hardware_control_available = (
            self.exposure_supported or self.gain_supported or self.brightness_supported)

        print(f"  frame-verified: exposure={'OK' if self.exposure_supported else 'UNSUPPORTED'}  "
              f"gain={'OK' if self.gain_supported else 'UNSUPPORTED'}  "
              f"brightness={'OK' if self.brightness_supported else 'UNSUPPORTED'}")
        if not self.hardware_control_available:
            print("  WARNING: this camera driver accepts EXPOSURE/GAIN/BRIGHTNESS writes but "
                  "the actual video never got measurably brighter/darker when tested - it's "
                  "likely just echoing values back without the sensor changing. Automatic (and "
                  "manual key/slider) exposure control has no real effect on this device; "
                  "falling back entirely on the software brightness compensation.")

    def _clamped_set(self, cap, prop, new_val, baseline, max_drift):
        low, high = baseline - max_drift, baseline + max_drift
        new_val = max(low, min(new_val, high))
        cap.set(prop, new_val)
        return cap.get(prop), low, high

    def _adjust_property(self, cap, prop, direction, step, prop_name):
        """Applies one bounded step to a single camera property, tracks
        whether it's pinned at its drift-band edge, and re-centers the
        baseline if it's been pinned there for a long stretch (see class
        docstring, point 6). Returns (moved, applied_value) - moved is
        False when the requested step didn't actually change the reported
        value (already clamped to the nearest edge), so callers can
        cascade to the next lever exactly as before."""
        baseline_attr = f"baseline_{prop_name}"
        max_drift_attr = f"{prop_name}_max_drift"
        baseline = getattr(self, baseline_attr)
        max_drift = getattr(self, max_drift_attr)

        cur = cap.get(prop)
        applied, low, high = self._clamped_set(cap, prop, cur + direction * step, baseline, max_drift)
        moved = abs(applied - cur) > 0.05

        at_edge = (direction > 0 and applied >= high - 1e-6) or (direction < 0 and applied <= low + 1e-6)
        if at_edge:
            self._pinned_checks[prop_name] += 1
            if self._pinned_checks[prop_name] >= self.PINNED_CHECKS_BEFORE_REBASELINE:
                shift = direction * max_drift * 0.8
                setattr(self, baseline_attr, baseline + shift)
                self._pinned_checks[prop_name] = 0
                print(f"  [{prop_name}] pinned at drift limit for a while - re-centering "
                      f"baseline by {shift:+.2f} to free up headroom")
        else:
            self._pinned_checks[prop_name] = 0

        return moved, applied

    def reset_to_baseline(self, cap):
        if self.baseline_exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.baseline_exposure)
            cap.set(cv2.CAP_PROP_GAIN, self.baseline_gain)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self.baseline_brightness)
            self._error_integral = 0.0
            self.last_status = "reset to baseline (frame went black)"

    def force_toward_dark_end(self, cap):
        """Snaps every supported property straight to the DARKEST edge of
        its own drift band, instead of continuing the gradual per-check
        proportional step. Mirrors reset_to_baseline's near-black safety
        net: a sustained near-white reading means the gradual correction
        isn't keeping up (or is being fought by something else), so a
        hard, immediate correction is used instead of creeping toward it
        one small step at a time."""
        if self.exposure_supported and self.baseline_exposure is not None:
            self._clamped_set(cap, cv2.CAP_PROP_EXPOSURE,
                               self.baseline_exposure - self.exposure_max_drift,
                               self.baseline_exposure, self.exposure_max_drift)
        if self.gain_supported and self.baseline_gain is not None:
            self._clamped_set(cap, cv2.CAP_PROP_GAIN,
                               self.baseline_gain - self.gain_max_drift,
                               self.baseline_gain, self.gain_max_drift)
        if self.brightness_supported and self.baseline_brightness is not None:
            self._clamped_set(cap, cv2.CAP_PROP_BRIGHTNESS,
                               self.baseline_brightness - self.brightness_max_drift,
                               self.baseline_brightness, self.brightness_max_drift)
        self._error_integral = 0.0
        self.last_status = "forced toward minimum (frame stayed near-white)"

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
        # Some UVC webcam drivers silently revert CAP_PROP_AUTO_EXPOSURE
        # back to their OWN firmware auto-exposure after a while (or after
        # certain events), even though set_baseline() forced it to manual
        # once at startup. When that happens, the camera's own auto
        # exposure fights whatever this loop - or the manual Exposure/
        # Brightness sliders - try to set, and exposure can drift/stay too
        # bright no matter what the software does; this was never
        # re-checked after startup. Re-asserting the same manual value
        # periodically (BEFORE the manual_lock/paused checks below - the
        # sliders need this forced too, to have any real effect at all) is
        # a cheap, harmless guard, and prints clearly whenever it actually
        # had to correct something, so this failure mode is directly
        # diagnosable instead of just looking like "the software logic is
        # broken".
        if self._auto_exposure_manual_value is not None:
            recheck_now = time.time()
            if (recheck_now - self._last_manual_reassert) > 3.0:
                self._last_manual_reassert = recheck_now
                current_ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                # Only warn on a genuine TRANSITION (compared to the last
                # OBSERVED reading, not the originally-requested value) -
                # on drivers where readback never confirmed the manual
                # value in the first place (see set_baseline's
                # _ae_readback_reliable), every check would otherwise
                # "mismatch" forever and print a false "reverted to auto"
                # warning every single cycle even though nothing ever
                # actually changed. Still write the value defensively
                # either way - harmless, and a real safety net on drivers
                # where readback DOES work.
                if (self._ae_readback_reliable and self._last_observed_ae is not None
                        and abs(current_ae - self._last_observed_ae) > 0.01
                        and abs(current_ae - self._auto_exposure_manual_value) > 0.01):
                    print(f"  [exposure] AUTO_EXPOSURE changed to {current_ae:.2f} "
                          f"(expected {self._auto_exposure_manual_value:.2f}) - camera "
                          f"firmware likely reverted to its own auto mode; re-asserting manual.")
                self._last_observed_ae = current_ae
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._auto_exposure_manual_value)

        if self.manual_lock:
            # Fully hands-off: the Controls sliders are driving exposure/
            # brightness directly elsewhere in the main loop.
            return

        if not self.hardware_control_available:
            self.last_status = "no hardware exposure control - relying on software compensation"
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

        # Symmetric safety net for the opposite extreme: a run of
        # consecutive near-white (blown-out) frames means the gradual
        # per-check correction isn't recovering fast enough - force a
        # hard snap toward the dark end instead of continuing to creep.
        if mean_brightness > 247.0:
            self._white_frame_streak += 1
            if self._white_frame_streak >= 3:
                self.force_toward_dark_end(cap)
                self._white_frame_streak = 0
            return
        else:
            self._white_frame_streak = 0

        self.frame_count += 1
        if self.frame_count % self.check_interval != 0:
            return

        mean = mean_brightness

        if self.TARGET_LOW <= mean <= self.TARGET_HIGH:
            self.last_status = f"stable (leaf mean {mean:.0f})"
            self._error_integral *= 0.5  # decay faster once in-band, avoid stale windup
            return

        error = self.TARGET_MID - mean
        # Leaky integral: a fading memory of recent error so a PERSISTENT
        # miss pushes harder over time, without one old extreme value
        # dominating forever (clamped, i.e. anti-windup).
        self._error_integral = max(-200.0, min(self._error_integral * 0.85 + error, 200.0))

        proportional = max(-1.0, min(error / 40.0, 1.0))
        combined = max(-1.0, min(proportional + self.integral_gain * (self._error_integral / 200.0), 1.0))
        direction = 1.0 if combined >= 0 else -1.0
        step_frac = min(abs(combined), self.max_step)

        exposure_step = step_frac * self.exposure_max_drift * self.exposure_step_frac
        gain_step = step_frac * self.gain_max_drift * self.gain_step_frac
        brightness_step = step_frac * self.brightness_max_drift * self.brightness_step_frac
        moved = False

        # BUG FIX: gain was previously only ever RAISED (direction > 0),
        # never lowered - so if gain had been pushed up at any point (a
        # dim room, a startup default), there was no code path that ever
        # brought it back down again once the scene needed to be darker.
        # That's a direct, concrete cause of "exposure stays too high":
        # gain keeps amplifying the signal regardless of what exposure/
        # brightness are doing. Tried FIRST when darkening (lowering gain
        # has no image-quality downside, unlike raising it, so there's no
        # reason to prefer exposure/shutter here the way there is when
        # brightening).
        if direction < 0 and self.gain_supported:
            moved, _ = self._adjust_property(cap, cv2.CAP_PROP_GAIN, direction, gain_step, "gain")

        if not moved and self.exposure_supported:
            moved, _ = self._adjust_property(
                cap, cv2.CAP_PROP_EXPOSURE, direction, exposure_step, "exposure")

        if not moved and direction > 0 and self.gain_supported:
            moved, _ = self._adjust_property(cap, cv2.CAP_PROP_GAIN, direction, gain_step, "gain")

        if not moved and self.brightness_supported:
            self._adjust_property(cap, cv2.CAP_PROP_BRIGHTNESS, direction, brightness_step, "brightness")

        word = "raising" if direction > 0 else "lowering"
        self.last_status = f"{word} (leaf mean {mean:.0f})"


def auto_gamma_correct(bgr, target_mean=140.0, tolerance=12.0, smoothed_mean=None):
    """Local shadow-lifting (CLAHE on L-channel) so a bright/uneven
    background doesn't hide a dark subject, followed by a LINEAR
    multiplicative brightness scale toward target_mean.

    A gamma/power-law curve (the previous approach here) has almost no
    effect near the ends of the 0-255 range: for a pixel already at 230,
    (230/255)^n stays close to 230/255 for nearly any n, so a genuinely
    overexposed frame barely moves even with the correction pointed the
    right direction - the picture can stay blown-out white regardless.
    A flat multiplicative scale (output = input * scale) has no such
    dead zone: it pulls EVERY pixel, including near-white ones,
    proportionally toward target. It can't recover pixels that are
    already fully clipped to 255 (no software post-process can - only
    reducing the camera's real exposure prevents clipping in the first
    place), but it does pull the rest of the frame - including the
    leaf's own midtones - into a usable range instead of leaving them
    pinned near white.

    smoothed_mean, when given, is an EMA of this reading from prior
    frames (see gamma_ema_alpha) and is what actually drives the
    correction strength - a single noisy or clutter-confused frame's raw
    mean can no longer yank it around and read as a brightness "flash".
    The function still returns this frame's raw (un-smoothed) mean as
    raw_mean so the caller can fold it into the EMA for the next call."""
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


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def create_rembg_session(cfg: Config):
    """Call once at startup (never per-frame). Returns None - and prints
    why - if rembg isn't installed or the model fails to load, in which
    case refine_bbox_with_rembg() below becomes a no-op and every save
    path silently falls back to the contour method's own bbox, exactly
    like before this feature existed."""
    if not cfg.use_rembg_refinement:
        return None
    if not REMBG_AVAILABLE:
        print("  [rembg] package not installed - capture-time crop refinement disabled "
              "(pip install \"rembg[cpu]\" to enable it). Everything else is unaffected.")
        return None
    assert _rembg_new_session is not None  # guaranteed by REMBG_AVAILABLE above
    try:
        session = _rembg_new_session(cfg.rembg_model)
        print(f"  [rembg] crop refinement ready (model={cfg.rembg_model})")
        return session
    except Exception as e:
        print(f"  [rembg] failed to load model ({e}) - crop refinement disabled")
        return None


def refine_bbox_with_rembg(frame_bgr, fallback_bbox, session, cfg: Config):
    """Runs rembg ONCE on the frame being saved to get a class-agnostic
    salient-object mask, and uses it in place of the contour method's
    bbox if - and only if - what it found looks plausibly like the leaf
    (green-dominant; see rembg_color_margin_floor). rembg has no notion
    of "leaf" at all, so this plausibility gate is what stops it from
    confidently handing back some other salient object in frame (a hand,
    a reflection, a shadow) instead.

    Only ever called at the moment of saving, not per-frame - this is
    what keeps its ~0.7-1.5 FPS speed from touching the live preview.
    Returns fallback_bbox unchanged if session is None (rembg unavailable
    or disabled), if rembg finds nothing, or if what it found doesn't
    pass the color-plausibility gate."""
    if session is None:
        return fallback_bbox
    assert _rembg_remove is not None  # session is only ever non-None when rembg is available

    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = _rembg_remove(rgb, session=session, only_mask=True)
        # only_mask=True + an ndarray input always yields an ndarray at
        # runtime - remove()'s declared return type is a broad Union
        # because it also supports bytes/PIL-Image inputs/outputs.
        assert isinstance(mask, np.ndarray)
    except Exception as e:
        print(f"  [rembg] refinement failed this capture ({e}) - using contour crop")
        return fallback_bbox

    _, binary = cv2.threshold(mask, cfg.rembg_mask_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_bbox

    largest = max(contours, key=cv2.contourArea)
    region_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(region_mask, [largest], -1, 255, thickness=cv2.FILLED)

    means = _mean_bgr_in_mask(frame_bgr, region_mask)
    if means is None:
        return fallback_bbox
    b, g, r = means
    color_margin = min(g - r, g - b)
    if color_margin <= cfg.rembg_color_margin_floor:
        return fallback_bbox  # not green - don't trust it over the contour box

    x, y, w, h = cv2.boundingRect(largest)
    candidate_bbox = (x, y, w, h)
    # rembg is class-agnostic salient-object detection with NO notion of
    # "leaf" - `largest` is just whatever single object it found most
    # visually prominent in the ENTIRE frame, which can be a completely
    # different object (a knob, a clip, a reflection) than the leaf the
    # contour detector was actually tracking. The color-margin gate above
    # only checks "is this greenish", which a weakly-lit object can clear
    # by chance - it does NOT check "is this the same object". Confirmed
    # live: the on-screen box was correctly on the leaf, but the saved
    # photo was a completely unrelated round object elsewhere in frame,
    # because rembg's pick barely passed the color gate with no positional
    # relationship to what was being tracked. Requiring real overlap with
    # the contour bbox is what makes this a REFINEMENT of the tracked
    # object's edges, rather than a silent replacement with something else.
    if fallback_bbox is not None and _bbox_iou(candidate_bbox, fallback_bbox) < 0.15:
        return fallback_bbox  # not the same object - don't trust it over the contour box

    print(f"  [rembg] refined crop: ({x},{y},{w},{h})"
          + (f"  (contour had {fallback_bbox})" if fallback_bbox is not None else "  (contour found nothing)"))
    return candidate_bbox


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


def show_refining_banner(display, window_name, msg="Refining crop..."):
    """rembg (see refine_bbox_with_rembg, and the periodic live-tracker
    supervision in main()) blocks for roughly 0.3-1.5s - long enough that
    the preview would otherwise look frozen/hung. This draws a quick
    banner and flushes it to screen BEFORE the blocking call, so the
    pause reads as "the app is doing something" rather than "the app
    broke"."""
    banner = display.copy()
    h, w = banner.shape[:2]
    (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    tx, ty = (w - mw) // 2, h // 2
    cv2.rectangle(banner, (tx - 24, ty - mh - 18), (tx + mw + 24, ty + 16), (0, 0, 0), -1)
    cv2.putText(banner, msg, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_WARNING, 2, cv2.LINE_AA)
    cv2.imshow(window_name, banner)
    cv2.waitKey(1)


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


def draw_debug_stats(display, mask_fraction, contour_fraction, exposure_brightness, score):
    """Small text readout, directly above the debug mask inset ('m'
    toggles both together): the four numbers that matter most for
    diagnosing "is my mask eating the background" at a glance.
      mask%     - how much of the FRAME the raw per-pixel detection mask
                  covers, before any contour is picked. High here even
                  when contour% looks reasonable is the first sign the
                  color threshold itself is too inclusive.
      contour%  - how much of the frame the SELECTED (winning) contour
                  covers - this is what actually becomes the bounding
                  box and the saved crop.
      exposure  - the (decontaminated) brightness reading actually being
                  fed to the auto-exposure controller this frame - watch
                  this to see whether exposure hunting tracks a sane
                  number or something contaminated/wrong.
      score     - the winning candidate's raw score from
                  _score_candidates, for comparing against what a
                  competing candidate would have scored."""
    lines = [
        f"mask: {mask_fraction * 100:5.1f}%   contour: {contour_fraction * 100:5.1f}%",
        f"exposure sample: {exposure_brightness:5.1f}   score: {score:5.2f}",
    ]
    h, w = display.shape[:2]
    inset_h = 150  # matches draw_debug_mask_inset's inset_h - panel sits directly above it
    panel_h = 24 * len(lines) + 12
    x1, y1 = 16, h - inset_h - 16 - panel_h - 8
    x2, y2 = x1 + 320, y1 + panel_h
    overlay = display.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (x1 + 10, y1 + 24 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_TEXT_PRIMARY, 1, cv2.LINE_AA)


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

    if (exposure_ctrl.manual_lock and exposure_ctrl.baseline_exposure is not None
            and exposure_ctrl.baseline_brightness is not None):
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
    rembg_session = create_rembg_session(cfg)
    last_live_rembg_check = 0.0  # wall-clock time of the last periodic live-tracker supervision
    adaptive = AdaptiveThresholds()
    tracker = LeafTracker(cfg.bbox_smooth_alpha, cfg.bbox_hold_frames)
    prev_bbox = None  # last tracked box, fed back in as a continuity hint
    zoom_lost_since = None  # wall-clock time the leaf was last seen, for auto-zoom's reset-to-wide-view
    # Stall guard for auto-zoom (see Config.auto_zoom_stall_timeout_sec):
    # tracks whether area_ratio is actually improving while zoom pushes in
    # a given direction, so a non-converging zoom-in doesn't run forever.
    zoom_stall_direction = 0        # +1 zooming in, -1 zooming out, 0 idle
    zoom_stall_reference_ratio = None
    zoom_stall_since = None
    exposure_mean_ema = None  # EMA of the exposure-driving brightness reading
    gamma_mean_ema = None     # EMA of the gamma-driving brightness reading
    frame_idx = 0
    prev_leaf_found = False
    frames_since_transition = cfg.brightness_transition_hold_frames  # start "settled"

    refine_active = False
    refine_frames_left = 0
    refine_best_frame, refine_best_bbox, refine_best_score = None, None, -1.0

    # Frozen "last real measurement" - reused during a brief bridged
    # detection dropout (see LeafTracker hold_frames) so a 1-4 frame gap
    # doesn't read as "no leaf" and doesn't reset stability/presence.
    last_known_brightness = last_known_sharpness = last_known_vein_score = 0.0
    last_known_area_ratio = 0.0

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
        # pan_x/pan_y are owned by the auto-zoom-toward-leaf logic below.
        raw_frame = apply_digital_zoom(raw_frame_full, cfg.zoom_factor, cfg.pan_x, cfg.pan_y)
        # Correct any global color cast (e.g. cyan/teal tint) before the
        # frame is used for anything else - detection, exposure sampling,
        # and gamma correction all assume roughly neutral color.
        raw_frame = gray_world_white_balance(raw_frame)

        raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        frame, raw_gamma_mean = auto_gamma_correct(raw_frame, smoothed_mean=gamma_mean_ema)
        gamma_mean_ema = raw_gamma_mean if gamma_mean_ema is None else (
            cfg.gamma_ema_alpha * raw_gamma_mean + (1 - cfg.gamma_ema_alpha) * gamma_mean_ema)
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_idx += 1
        # Switch hysteresis only kicks in once the SAME track has been
        # confirmed over several consecutive frames (so an early bad lock
        # can't reinforce itself), and is periodically forced off for one
        # "cold" comparison so a better candidate elsewhere in frame can
        # still win even against a long-held track. See detect_leaf().
        was_confirmed = tracker.confirmed_frames >= cfg.continuity_min_confirmed_frames
        cold_recheck = (frame_idx % cfg.cold_recheck_interval_frames == 0)
        apply_hysteresis = was_confirmed and not cold_recheck

        leaf_mask, bbox, area_ratio, detect_debug = detect_leaf(
            frame, cfg, prev_bbox=prev_bbox, apply_hysteresis=apply_hysteresis,
            debug=cfg.show_debug_mask)
        leaf_found = bbox is not None

        # --- Auto-zoom toward the leaf --------------------------------------
        # Adjusts cfg.zoom_factor/pan_x/pan_y for the NEXT frame
        # (apply_digital_zoom runs at the top of the loop, one frame ahead -
        # a normal visual-servoing lag). Goal: narrow the field of view onto
        # the leaf BEFORE relying on the detector for anything else, so a
        # wide/cluttered shot (background, tripod, other objects) gets
        # cropped away and only a clean, leaf-dominated view ever reaches the
        # detection/quality/capture logic below - exactly like a person
        # framing a photo by hand before deciding it looks right.
        #
        # Reacts to the tracker-CONFIRMED detection (held for at least
        # continuity_min_confirmed_frames consecutive frames - the same
        # bar hysteresis uses elsewhere, see LeafTracker/Config), not the
        # raw per-frame bbox. A fleeting 1-2 frame false positive (a dark
        # object briefly clearing the color/shape gates) can no longer
        # yank pan/zoom toward it before the tracker has proven the lock
        # is real; a genuine leaf still confirms within well under a
        # second, so this costs negligible responsiveness.
        zoom_track_confirmed = tracker.confirmed_frames >= cfg.continuity_min_confirmed_frames
        if leaf_found and zoom_track_confirmed:
            zoom_lost_since = None
            zx, zy, zw, zh = bbox
            leaf_cx, leaf_cy = zx + zw / 2.0, zy + zh / 2.0
            zframe_h, zframe_w = frame.shape[:2]
            err_x = (leaf_cx - zframe_w / 2.0) / zframe_w
            err_y = (leaf_cy - zframe_h / 2.0) / zframe_h
            centered = (abs(err_x) <= cfg.auto_zoom_center_tol
                        and abs(err_y) <= cfg.auto_zoom_center_tol)
            # Deliberately looser than `centered` - see
            # Config.auto_zoom_safe_to_zoom_tol for why zoom must NOT be
            # gated behind the tight centering check.
            safe_to_zoom = (abs(err_x) <= cfg.auto_zoom_safe_to_zoom_tol
                            and abs(err_y) <= cfg.auto_zoom_safe_to_zoom_tol)
            well_sized = cfg.auto_zoom_target_low <= area_ratio <= cfg.auto_zoom_target_high

            # Pan and zoom are independent checks (not if/elif) - both can
            # run in the same frame. Without this, zoom could never take
            # its first step whenever the leaf started out more than
            # auto_zoom_center_tol off-center, since panning has no visual
            # effect at all until zoom_factor rises above 1.0 (a deadlock;
            # see Config.auto_zoom_safe_to_zoom_tol).
            if not centered:
                step = cfg.auto_pan_step
                cfg.pan_x = float(np.clip(
                    cfg.pan_x + np.clip(err_x, -step, step) / cfg.zoom_factor, 0.0, 1.0))
                cfg.pan_y = float(np.clip(
                    cfg.pan_y + np.clip(err_y, -step, step) / cfg.zoom_factor, 0.0, 1.0))

            if well_sized:
                # Reached the target - clear any stall tracking so a
                # FUTURE need to zoom (leaf moves away, etc.) starts fresh.
                zoom_stall_direction, zoom_stall_reference_ratio, zoom_stall_since = 0, None, None
            elif not safe_to_zoom:
                # Too far off-center to safely zoom this frame - not a
                # stall (panning is still actively correcting), just skip.
                pass
            else:
                direction = 1 if area_ratio < cfg.auto_zoom_target_low else -1
                zoom_now = time.time()
                if direction != zoom_stall_direction or zoom_stall_reference_ratio is None:
                    # Direction just changed (or this is the first push) -
                    # start a fresh reference point to measure progress
                    # against.
                    zoom_stall_direction = direction
                    zoom_stall_reference_ratio = area_ratio
                    zoom_stall_since = zoom_now
                elif abs(area_ratio - zoom_stall_reference_ratio) >= cfg.auto_zoom_stall_improve_eps:
                    # Real progress since the last checkpoint - reset the
                    # clock rather than accumulating stall time forever.
                    zoom_stall_reference_ratio = area_ratio
                    zoom_stall_since = zoom_now

                stalled = (zoom_stall_since is not None
                           and (zoom_now - zoom_stall_since) > cfg.auto_zoom_stall_timeout_sec)
                if not stalled:
                    if direction > 0:
                        cfg.zoom_factor = min(cfg.zoom_factor + cfg.auto_zoom_step, cfg.zoom_max)
                    else:
                        cfg.zoom_factor = max(cfg.zoom_factor - cfg.auto_zoom_step, cfg.zoom_min)
                # else: this direction hasn't actually improved area_ratio
                # in a while (e.g. the leaf drifting toward the crop edge
                # as fast as zoom tightens) - hold zoom where it is instead
                # of continuing to push it further with nothing to show
                # for it.

            # Keep the Controls window's Zoom slider showing the live
            # auto-zoom value, so it doesn't look frozen/stuck to a user
            # watching it. read_controls() reads this same slider back at
            # the top of next frame's loop - but since it's now showing
            # the value auto-zoom just set, that read-back is a no-op;
            # auto-zoom then adjusts further from there based on the next
            # frame's own detection. No feedback loop, just harmless
            # round-tripping while a leaf is being actively tracked.
            cv2.setTrackbarPos(SLIDER_ZOOM, CONTROLS_WIN, int(round(cfg.zoom_factor * 100)))
        elif leaf_found:
            # Found, but the tracker hasn't confirmed it yet (normal for
            # the first fraction of a second of any fresh lock) - hold
            # pan/zoom steady and do NOT count this as "lost" below, or a
            # slow-to-confirm lock could spuriously reset zoom back to a
            # wide view before it ever got a real chance to act.
            zoom_lost_since = None
        else:
            zoom_now = time.time()
            if zoom_lost_since is None:
                zoom_lost_since = zoom_now
            elif (zoom_now - zoom_lost_since) > cfg.auto_zoom_lost_reset_sec:
                # Leaf has been missing for a while - give up and reset to a
                # neutral wide view to search again, rather than staying
                # zoomed in on empty space (or clutter) indefinitely.
                cfg.zoom_factor = cfg.zoom_min
                cfg.pan_x, cfg.pan_y = 0.5, 0.5
        # ---------------------------------------------------------------------

        # display_bbox is the smoothed/held box (see LeafTracker), fed back
        # in as next frame's continuity hint.
        display_bbox = tracker.update(bbox)
        prev_bbox = display_bbox

        # Periodic live supervision (see Config.rembg_live_supervision):
        # the fast contour tracker can lock onto the wrong object and then
        # (via switch hysteresis) actively defend that lock instead of
        # self-correcting. Every rembg_live_recheck_interval_sec, consult
        # rembg once on THIS frame and force-reseed the tracker if it
        # disagrees with what's currently shown on screen (or if nothing
        # is currently tracked at all) - this is what fixes the on-screen
        # box, not just the final saved crop (see refine_bbox_with_rembg,
        # which only ever runs at the moment of saving).
        rembg_check_now = time.time()
        if (rembg_session is not None and cfg.rembg_live_supervision
                and (rembg_check_now - last_live_rembg_check) > cfg.rembg_live_recheck_interval_sec):
            last_live_rembg_check = rembg_check_now
            show_refining_banner(display, window_name, msg="Verifying leaf lock...")
            rembg_hint = refine_bbox_with_rembg(frame, None, rembg_session, cfg)
            tracker_agrees = (display_bbox is not None and rembg_hint is not None
                               and _bbox_iou(rembg_hint, display_bbox) > 0.3)
            if rembg_hint is not None and not tracker_agrees:
                print(f"  [rembg] live supervision: re-seeding tracker "
                      f"(on-screen box was {display_bbox}, rembg says {rembg_hint})")
                tracker.smoothed_bbox = None
                tracker.confirmed_frames = 0
                tracker.frames_since_seen = 0
                display_bbox = tracker.update(rembg_hint)
                prev_bbox = display_bbox

        # Once a track is confirmed, the smoothed/held box is trustworthy
        # enough to drive MEASUREMENT and the SAVED crop too, not just the
        # on-screen display - otherwise raw per-frame jitter still reaches
        # the saved image even though the drawn box looks calm. Before
        # confirmation, use the raw detection so a fresh lock isn't
        # laggily smoothed away. The smoothed box is only trusted when it
        # still substantially overlaps THIS frame's raw detection - if the
        # two have diverged (e.g. the smoothed box is a stale lock on the
        # wrong object), fall back to raw immediately rather than letting
        # measurement/exposure keep chasing a stale region indefinitely.
        track_confirmed = tracker.confirmed_frames >= cfg.continuity_min_confirmed_frames
        smoothed_agrees = (display_bbox is not None and bbox is not None
                            and _bbox_iou(display_bbox, bbox) > 0.3)
        measurement_bbox = display_bbox if (track_confirmed and smoothed_agrees) else bbox
        if measurement_bbox is None:
            measurement_bbox = display_bbox  # bridged gap - use the held position for cropping

        # leaf_present is broader than leaf_found: it stays True while the
        # tracker is still HOLDING a box on screen (see LeafTracker
        # hold_frames), bridging brief 1-4 frame detection dropouts (a
        # momentary contour break, one noisy MJPG frame) that are normal
        # and shouldn't read as "no leaf" or reset the stability/presence
        # timers that gate auto-capture - previously they did, which is
        # very likely why auto-capture rarely fired even with a leaf
        # sitting still in frame for a long time.
        leaf_present = display_bbox is not None

        # Exposure target: use the LEAF's own brightness (in the raw,
        # pre-gamma frame) when one is visible, whole-frame mean otherwise.
        #
        # Fix #4 (contaminated-mask exposure): if the "leaf" mask has
        # fused in some background (see detect_leaf), a plain mean over
        # every masked pixel bakes that contamination straight into the
        # exposure target - a dark corner of background mixed in can
        # read as "too dark" and drive exposure/gain up even while the
        # actual leaf is fine or already blown out. Two independent
        # guards against that:


        
        #   1. Erode the mask before sampling - contamination is
        #      concentrated at the boundary where a merged blob fused
        #      with the background, so pixels solidly inside the
        #      contour are far more likely to be genuine leaf.
        #   2. Blend the masked reading toward the whole-frame reading
        #      as area_ratio grows - a suspiciously large accepted
        #      region is itself a contamination red flag (same signal
        #      fix #1's size-saturated scoring reacts to), so it earns
        #      less trust rather than being taken at face value.
        if leaf_found:
            assert measurement_bbox is not None
            x, y, w, h = measurement_bbox
            x2b, y2b = min(x + w, raw_gray.shape[1]), min(y + h, raw_gray.shape[0])
            roi_raw = raw_gray[y:y2b, x:x2b]
            roi_mask_raw = leaf_mask[y:y2b, x:x2b]
            if roi_raw.size > 0 and np.any(roi_mask_raw):
                erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                eroded_mask = cv2.erode(roi_mask_raw, erode_kernel, iterations=1)
                sample_mask = eroded_mask if np.any(eroded_mask) else roi_mask_raw

                masked_mean = float(np.mean(roi_raw[sample_mask > 0]))
                whole_frame_mean = float(np.mean(raw_gray))
                size_trust = 1.0 - 0.6 * min(area_ratio / cfg.max_area_ratio, 1.0)
                exposure_mean = size_trust * masked_mean + (1.0 - size_trust) * whole_frame_mean
            else:
                exposure_mean = float(np.mean(raw_gray))
        else:
            exposure_mean = float(np.mean(raw_gray))

        # The reading source flips between whole-frame and leaf-only ROI
        # right on a found/not-found transition - use a slower EMA alpha
        # for a short window after that flip so the jump itself is
        # smoothed, not just ordinary per-frame noise (see A6 in Config).
        if leaf_found != prev_leaf_found:
            frames_since_transition = 0
        else:
            frames_since_transition += 1
        prev_leaf_found = leaf_found
        active_brightness_alpha = (cfg.brightness_transition_alpha
                                    if frames_since_transition < cfg.brightness_transition_hold_frames
                                    else cfg.brightness_ema_alpha)

        exposure_mean_ema = exposure_mean if exposure_mean_ema is None else (
            active_brightness_alpha * exposure_mean + (1 - active_brightness_alpha) * exposure_mean_ema)
        exposure_ctrl.maybe_adjust(cap, exposure_mean_ema)

        if leaf_found and prev_gray_full is not None:
            assert measurement_bbox is not None
            x, y, w, h = measurement_bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            cur_roi_m = gray_full[y:y2, x:x2]
            prev_roi_m = prev_gray_full[y:y2, x:x2]
            if cur_roi_m.shape == prev_roi_m.shape and cur_roi_m.size > 0:
                cur_norm = cur_roi_m.astype(np.float32) - float(np.mean(cur_roi_m))
                prev_norm = prev_roi_m.astype(np.float32) - float(np.mean(prev_roi_m))
                motion = float(np.mean(np.abs(cur_norm - prev_norm)))
            else:
                motion = 999.0
        elif leaf_present:
            # Bridging a brief dropout - nothing physically moves
            # meaningfully in a ~1-4 frame gap, so treat motion as
            # negligible rather than the "definitely moving" fallback
            # used for a genuine, sustained absence.
            motion = 0.0
        else:
            motion = 999.0
        prev_gray_full = gray_full

        brightness = sharpness = vein_score = 0.0

        if leaf_found:
            assert measurement_bbox is not None
            x, y, w, h = measurement_bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            gray_roi = gray_full[y:y2, x:x2]
            roi_mask = leaf_mask[y:y2, x:x2]

            brightness = compute_brightness(gray_roi, roi_mask)
            sharpness = compute_sharpness(gray_roi, roi_mask)
            vein_score = compute_vein_score(gray_roi, roi_mask)

            if cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio \
                    and cfg.min_brightness <= brightness <= cfg.max_brightness:
                adaptive.update(sharpness, vein_score, motion)

            last_known_brightness, last_known_sharpness, last_known_vein_score = brightness, sharpness, vein_score
            last_known_area_ratio = area_ratio
        elif leaf_present:
            # Reuse the last real measurement instead of zeroing out, so a
            # brief bridged gap doesn't read as "no leaf" to guidance and
            # doesn't reset the stability/presence tracking below.
            brightness, sharpness, vein_score = last_known_brightness, last_known_sharpness, last_known_vein_score
            area_ratio = last_known_area_ratio

        # Bounding box is drawn from the smoothed/held box whenever one is
        # available - a thick outline, corner accents, and a label, so it
        # can't be missed or blend into the background, and doesn't visibly
        # flicker on a one-frame detection dropout in clutter.
        if display_bbox is not None:
            draw_leaf_bbox(display, display_bbox)

        # Visualize the fixed exclusion zone (see Config.ignore_bottom_frac)
        # so it's directly visible/verifiable against the actual rig,
        # rather than a number that's hard to judge without seeing it.
        if cfg.ignore_bottom_frac > 0:
            excl_y = int(display.shape[0] * (1.0 - cfg.ignore_bottom_frac))
            cv2.line(display, (0, excl_y), (display.shape[1], excl_y), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "excluded from detection below this line",
                        (16, excl_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

        now = time.time()

        # Progressive floor relaxation: the longer this leaf has sat
        # present without a capture, the more the hard sharpness/vein
        # floor eases off (see Config.quality_floor_relax_frac) - this
        # keeps a floor that's simply uncalibrated for this camera/lens
        # from permanently blocking every capture, without giving up
        # quality checks immediately the way the hard timeout fallback
        # does. A leaf that just appeared still gets the full, strict
        # floor.
        presence_elapsed = (now - leaf_present_since) if leaf_present_since is not None else 0.0
        relax = min(presence_elapsed / cfg.capture_timeout_sec, 1.0)
        sharp_floor = cfg.sharpness_threshold * (1.0 - cfg.quality_floor_relax_frac * relax)
        vein_floor = cfg.vein_score_threshold * (1.0 - cfg.quality_floor_relax_frac * relax)

        sharp_thresh = adaptive.sharp_target(sharp_floor)
        vein_thresh = adaptive.vein_target(vein_floor)
        live_motion_threshold = adaptive.motion_target(cfg.motion_threshold)

        guidance_text, all_pass = decide_guidance(
            cfg, area_ratio, brightness, sharpness, vein_score, leaf_present,
            sharp_thresh, vein_thresh)

        score = quality_score(cfg, area_ratio, brightness, sharpness, vein_score) if leaf_present else 0.0

        if leaf_present:
            if leaf_present_since is None:
                # leaf_present just became True, which (see LeafTracker)
                # can only happen off a genuine fresh detection, so
                # leaf_found is guaranteed True here too.
                leaf_present_since = now
                captured_this_presence = False
                presence_best_frame, presence_best_bbox, presence_best_score = frame.copy(), measurement_bbox, score
            elif leaf_found and score > presence_best_score:
                presence_best_frame, presence_best_bbox, presence_best_score = frame.copy(), measurement_bbox, score
        else:
            leaf_present_since = None

        steady_now = all_pass and motion < live_motion_threshold
        stability_hist.append(steady_now)
        is_stable = len(stability_hist) == stability_hist.maxlen and all(stability_hist)

        if refine_active:
            # Stability window already completed once - keep evaluating a
            # short extra stretch and bank whichever frame in it scores
            # highest, rather than committing to the exact frame that
            # happened to complete the streak (see Config.capture_refine_frames).
            if steady_now:
                if score > refine_best_score:
                    refine_best_frame, refine_best_bbox, refine_best_score = frame.copy(), measurement_bbox, score
                refine_frames_left -= 1
                refine_done = refine_frames_left <= 0
            else:
                refine_done = True  # lost stability mid-window - bank the best seen so far
            if refine_done:
                if rembg_session is not None:
                    show_refining_banner(display, window_name)
                final_bbox = refine_bbox_with_rembg(refine_best_frame, refine_best_bbox, rembg_session, cfg)
                save_frame(refine_best_frame, cfg, refine_best_score, bbox=final_bbox)
                captures_count += 1
                _beep()
                last_capture_time = now
                flash_until = now + cfg.capture_flash_sec
                captured_this_presence = True
                stability_hist.clear()
                refine_active = False
        elif is_stable and (now - last_capture_time) > cfg.capture_cooldown_sec:
            refine_active = True
            refine_frames_left = cfg.capture_refine_frames
            refine_best_frame, refine_best_bbox, refine_best_score = frame.copy(), measurement_bbox, score
        elif (leaf_present_since is not None and not captured_this_presence
              and (now - leaf_present_since) > cfg.capture_timeout_sec
              and presence_best_frame is not None):
            low_confidence = presence_best_score < cfg.low_confidence_score
            if cfg.strict_reject_blurry and low_confidence:
                print("  [timeout] best frame seen is still below the quality floor - "
                      "continuing to wait instead of saving (strict_reject_blurry=True); "
                      "try moving/refocusing the leaf")
                leaf_present_since = now  # restart the timeout window rather than giving up
            else:
                if rembg_session is not None:
                    show_refining_banner(display, window_name)
                final_bbox = refine_bbox_with_rembg(presence_best_frame, presence_best_bbox, rembg_session, cfg)
                save_frame(presence_best_frame, cfg, presence_best_score,
                           bbox=final_bbox, low_confidence=low_confidence)
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
            
            track_str = f"confirmed({tracker.confirmed_frames})" if track_confirmed else \
                f"locking({tracker.confirmed_frames}/{cfg.continuity_min_confirmed_frames})"
            refine_str = f" refining({refine_frames_left})" if refine_active else ""
            print(f"[status] guidance='{guidance_text}'  score={score:.1f}  "
                  f"sharp={sharpness:.0f}/{sharp_thresh:.0f}  vein={vein_score:.1f}/{vein_thresh:.1f}  "
                  f"adaptive={ready_str}  mode={mode_str}  track={track_str}{refine_str}  "
                  f"zoom={cfg.zoom_factor:.2f}x  exposure={exposure_ctrl.last_status}")
            # Raw camera property readback, printed every status tick (not
            # just once at startup) - the fastest way to tell whether
            # "exposure stays too high" is a software logic issue (these
            # numbers should visibly move toward the baseline over a few
            # seconds) or the camera driver just not honoring writes at all
            # (these numbers never move no matter what the software does).
            print(f"  [camera] AUTO_EXPOSURE={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}  "
                  f"EXPOSURE={cap.get(cv2.CAP_PROP_EXPOSURE):.2f}  "
                  f"GAIN={cap.get(cv2.CAP_PROP_GAIN):.2f}  "
                  f"BRIGHTNESS={cap.get(cv2.CAP_PROP_BRIGHTNESS):.2f}  "
                  f"frame_mean={exposure_mean_ema:.1f}")
            last_status_print = now

        draw_overlay(display, guidance_text, score, leaf_present,
                     len(stability_hist), stability_hist.maxlen,
                     captures_count, now < flash_until)

        if cfg.show_debug_mask:
            debug_mask_fraction = detect_debug["raw_mask_fraction"] if detect_debug else 0.0
            debug_score = detect_debug["score"] if detect_debug else 0.0
            draw_debug_stats(display, debug_mask_fraction, area_ratio, exposure_mean_ema, debug_score)
            draw_debug_mask_inset(display, leaf_mask, leaf_found)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key != 0xFF:
            focus_title = get_foreground_window_title()
            char = chr(key) if 32 <= key < 127 else "?"
            print(f"[keypress] raw_code={key} char={char!r} focus_window={focus_title!r}")

        if key == ord('q'):
            break
        elif key == ord('s'):
            if rembg_session is not None:
                show_refining_banner(display, window_name)
            final_bbox = refine_bbox_with_rembg(frame, bbox, rembg_session, cfg)
            save_frame(frame, cfg, score, bbox=final_bbox, manual=True)
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