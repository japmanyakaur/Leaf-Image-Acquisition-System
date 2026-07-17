"""Digital zoom mechanics and the auto-zoom-toward-leaf servo logic.
Narrows the field of view onto the leaf BEFORE relying on the detector for
anything else, so a wide/cluttered shot gets cropped away and only a
clean, leaf-dominated view reaches quality/capture logic - the same as a
person framing a photo by hand before deciding it looks right."""

import time

import cv2
import numpy as np

from ui.controls_panel import CONTROLS_WIN, SLIDER_ZOOM


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


class AutoZoomController:
    """Owns cfg.zoom_factor/pan_x/pan_y while a leaf is being tracked.

    Reacts to the tracker-CONFIRMED detection (held for at least
    continuity_min_confirmed_frames consecutive frames - the same bar
    hysteresis uses elsewhere), not the raw per-frame bbox. A fleeting
    1-2 frame false positive (a dark object briefly clearing the
    color/shape gates) can't yank pan/zoom toward it before the tracker
    has proven the lock is real; a genuine leaf still confirms within
    well under a second.

    Pan and zoom are independent (not if/elif) - both can run in the
    same frame. apply_digital_zoom is a complete no-op while
    zoom_factor<=1.001, so panning has ZERO visible effect until zoom
    has already taken its first step; gating zoom behind the tight
    `centered` check would deadlock whenever the leaf starts out more
    than auto_zoom_center_tol off-center. Config.auto_zoom_safe_to_zoom_tol
    is a separate, looser tolerance used only to decide whether zoom is
    safe to run at all (avoids cropping the leaf out near a frame edge).

    A stall guard tracks whether area_ratio is actually improving while
    zoom pushes in a given direction; if it hasn't improved by
    auto_zoom_stall_improve_eps within auto_zoom_stall_timeout_sec,
    zoom holds where it is instead of continuing to push it further with
    nothing to show for it (e.g. the leaf drifting toward the crop edge
    as fast as zoom tightens)."""

    def __init__(self):
        self.zoom_lost_since = None  # wall-clock time the leaf was last seen
        self.zoom_stall_direction = 0        # +1 zooming in, -1 zooming out, 0 idle
        self.zoom_stall_reference_ratio = None
        self.zoom_stall_since = None

    def update(self, cfg, leaf_found, zoom_track_confirmed, bbox, area_ratio, frame_shape):
        """Adjusts cfg.zoom_factor/pan_x/pan_y for the NEXT frame
        (apply_digital_zoom runs at the top of the loop, one frame ahead -
        a normal visual-servoing lag)."""
        if leaf_found and zoom_track_confirmed:
            self.zoom_lost_since = None
            zx, zy, zw, zh = bbox
            leaf_cx, leaf_cy = zx + zw / 2.0, zy + zh / 2.0
            zframe_h, zframe_w = frame_shape[:2]
            err_x = (leaf_cx - zframe_w / 2.0) / zframe_w
            err_y = (leaf_cy - zframe_h / 2.0) / zframe_h
            centered = (abs(err_x) <= cfg.auto_zoom_center_tol
                        and abs(err_y) <= cfg.auto_zoom_center_tol)
            safe_to_zoom = (abs(err_x) <= cfg.auto_zoom_safe_to_zoom_tol
                            and abs(err_y) <= cfg.auto_zoom_safe_to_zoom_tol)
            well_sized = cfg.auto_zoom_target_low <= area_ratio <= cfg.auto_zoom_target_high

            if not centered:
                step = cfg.auto_pan_step
                cfg.pan_x = float(np.clip(
                    cfg.pan_x + np.clip(err_x, -step, step) / cfg.zoom_factor, 0.0, 1.0))
                cfg.pan_y = float(np.clip(
                    cfg.pan_y + np.clip(err_y, -step, step) / cfg.zoom_factor, 0.0, 1.0))

            if well_sized:
                # Reached the target - clear stall tracking so a FUTURE
                # need to zoom starts fresh.
                self.zoom_stall_direction, self.zoom_stall_reference_ratio, self.zoom_stall_since = 0, None, None
            elif not safe_to_zoom:
                # Too far off-center to safely zoom this frame - not a
                # stall (panning is still actively correcting), just skip.
                pass
            else:
                direction = 1 if area_ratio < cfg.auto_zoom_target_low else -1
                zoom_now = time.time()
                if direction != self.zoom_stall_direction or self.zoom_stall_reference_ratio is None:
                    # Direction just changed (or this is the first push) -
                    # start a fresh reference point to measure progress.
                    self.zoom_stall_direction = direction
                    self.zoom_stall_reference_ratio = area_ratio
                    self.zoom_stall_since = zoom_now
                elif abs(area_ratio - self.zoom_stall_reference_ratio) >= cfg.auto_zoom_stall_improve_eps:
                    # Real progress since the last checkpoint - reset the
                    # clock rather than accumulating stall time forever.
                    self.zoom_stall_reference_ratio = area_ratio
                    self.zoom_stall_since = zoom_now

                stalled = (self.zoom_stall_since is not None
                           and (zoom_now - self.zoom_stall_since) > cfg.auto_zoom_stall_timeout_sec)
                if not stalled:
                    if direction > 0:
                        cfg.zoom_factor = min(cfg.zoom_factor + cfg.auto_zoom_step, cfg.zoom_max)
                    else:
                        cfg.zoom_factor = max(cfg.zoom_factor - cfg.auto_zoom_step, cfg.zoom_min)

            # Keep the Controls window's Zoom slider showing the live
            # auto-zoom value, so it doesn't look frozen/stuck. Next
            # frame's read_controls() reads this same slider back, but
            # since it's now showing the value auto-zoom just set, that
            # read-back is a no-op - harmless round-tripping.
            cv2.setTrackbarPos(SLIDER_ZOOM, CONTROLS_WIN, int(round(cfg.zoom_factor * 100)))
        elif leaf_found:
            # Found, but the tracker hasn't confirmed it yet (normal for
            # the first fraction of a second of any fresh lock) - hold
            # pan/zoom steady and do NOT count this as "lost" below, or a
            # slow-to-confirm lock could spuriously reset zoom back to a
            # wide view before it ever got a real chance to act.
            self.zoom_lost_since = None
        else:
            zoom_now = time.time()
            if self.zoom_lost_since is None:
                self.zoom_lost_since = zoom_now
            elif (zoom_now - self.zoom_lost_since) > cfg.auto_zoom_lost_reset_sec:
                # Leaf has been missing for a while - give up and reset to
                # a neutral wide view to search again.
                cfg.zoom_factor = cfg.zoom_min
                cfg.pan_x, cfg.pan_y = 0.5, 0.5
