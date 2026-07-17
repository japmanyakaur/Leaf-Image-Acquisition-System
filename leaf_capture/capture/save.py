"""Saving captured photos, plus the optional rembg-based crop refinement
that runs ONCE at the moment of saving (never in the live per-frame loop -
rembg is far too slow for that, ~0.7-1.5 FPS vs ~9-10 FPS for the contour
detector)."""

import os
from datetime import datetime

import cv2
import numpy as np

# rembg is an OPTIONAL dependency. If it isn't installed, refinement is
# silently skipped and every save path falls back to the contour method's
# own bbox, exactly like before this feature existed.
try:
    from rembg import remove as _rembg_remove, new_session as _rembg_new_session
    REMBG_AVAILABLE = True
except ImportError:
    _rembg_remove = None
    _rembg_new_session = None
    REMBG_AVAILABLE = False

from config import Config
from localization.contour_localizer import _mean_bgr_in_mask, _bbox_iou


def create_rembg_session(cfg: Config):
    """Call once at startup (never per-frame). Returns None - and prints
    why - if rembg isn't installed or the model fails to load, in which
    case refine_bbox_with_rembg() below becomes a no-op."""
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
    (green-dominant) AND overlaps the contour method's own bbox. rembg
    has no notion of "leaf" at all - it's just whatever single object it
    found most visually prominent in the ENTIRE frame, which can be a
    completely different object (a knob, a clip, a reflection) than the
    leaf actually being tracked. The color-margin gate alone only checks
    "is this greenish", which a weakly-lit object can clear by chance -
    it does NOT check "is this the same object". Requiring real overlap
    with the contour bbox is what makes this a REFINEMENT of the tracked
    object's edges, rather than a silent replacement with something else.

    Only ever called at the moment of saving, not per-frame. Returns
    fallback_bbox unchanged if session is None, if rembg finds nothing,
    or if what it found doesn't pass the plausibility gates."""
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
