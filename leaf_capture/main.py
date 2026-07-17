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

import os
import time
import ctypes

import cv2
import numpy as np

from config import Config
from localization.contour_localizer import detect_leaf, _bbox_iou
from quality.quality_control import compute_brightness, compute_sharpness, compute_vein_score, AdaptiveThresholds
from tracking.leaf_tracker import LeafTracker
from tracking.auto_zoom import apply_digital_zoom, AutoZoomController
from capture.save import create_rembg_session, refine_bbox_with_rembg, save_frame, _beep
from capture.capture_decision import CaptureDecision
from camera.exposure_control import AutoExposureController, nudge_camera_property
from camera.camera_io import open_camera, auto_gamma_correct, gray_world_white_balance
from ui.overlay import (draw_leaf_bbox, draw_overlay, draw_debug_stats, draw_debug_mask_inset,
                         show_refining_banner)
from ui.controls_panel import create_controls_window, read_controls


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
    adaptive = AdaptiveThresholds()
    tracker = LeafTracker(cfg.bbox_smooth_alpha, cfg.bbox_hold_frames)
    auto_zoom = AutoZoomController()
    capture_decision = CaptureDecision(cfg, rembg_session)
    prev_bbox = None  # last tracked box, fed back in as a continuity hint
    last_live_rembg_check = 0.0  # wall-clock time of the last periodic live-tracker supervision
    exposure_mean_ema = None  # EMA of the exposure-driving brightness reading
    gamma_mean_ema = None     # EMA of the gamma-driving brightness reading
    frame_idx = 0
    prev_leaf_found = False
    frames_since_transition = cfg.brightness_transition_hold_frames  # start "settled"

    # Frozen "last real measurement" - reused during a brief bridged
    # detection dropout (see LeafTracker hold_frames) so a 1-4 frame gap
    # doesn't read as "no leaf" and doesn't reset stability/presence.
    last_known_brightness = last_known_sharpness = last_known_vein_score = 0.0
    last_known_area_ratio = 0.0

    window_name = "Leaf Capture System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    create_controls_window(cfg, window_name)

    prev_gray_full = None
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

        # Auto-zoom toward the leaf - adjusts cfg.zoom_factor/pan_x/pan_y
        # for the NEXT frame (apply_digital_zoom runs at the top of the
        # loop, one frame ahead - a normal visual-servoing lag).
        zoom_track_confirmed = tracker.confirmed_frames >= cfg.continuity_min_confirmed_frames
        auto_zoom.update(cfg, leaf_found, zoom_track_confirmed, bbox, area_ratio, frame.shape)

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
        # timers that gate auto-capture.
        leaf_present = display_bbox is not None

        # Exposure target: use the LEAF's own brightness (in the raw,
        # pre-gamma frame) when one is visible, whole-frame mean otherwise.
        #
        # If the "leaf" mask has fused in some background, a plain mean
        # over every masked pixel bakes that contamination straight into
        # the exposure target. Two independent guards against that:
        #   1. Erode the mask before sampling - contamination is
        #      concentrated at the boundary where a merged blob fused
        #      with the background, so pixels solidly inside the contour
        #      are far more likely to be genuine leaf.
        #   2. Blend the masked reading toward the whole-frame reading as
        #      area_ratio grows - a suspiciously large accepted region is
        #      itself a contamination red flag, so it earns less trust.
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
        # smoothed, not just ordinary per-frame noise.
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
        # so it's directly visible/verifiable against the actual rig.
        if cfg.ignore_bottom_frac > 0:
            excl_y = int(display.shape[0] * (1.0 - cfg.ignore_bottom_frac))
            cv2.line(display, (0, excl_y), (display.shape[1], excl_y), (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "excluded from detection below this line",
                        (16, excl_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

        now = time.time()

        guidance_text, score, sharp_thresh, vein_thresh = capture_decision.update(
            adaptive, now, frame, display, window_name,
            leaf_present, leaf_found, measurement_bbox,
            area_ratio, brightness, sharpness, vein_score, motion)

        if (now - last_status_print) > 1.0:
            ready_str = "learned" if adaptive.ready else f"learning ({len(adaptive.sharp_samples)}/{adaptive.min_samples})"
            mode_str = "MANUAL(locked)" if exposure_ctrl.manual_lock else (
                "paused(key)" if exposure_ctrl.paused else "auto")

            track_str = f"confirmed({tracker.confirmed_frames})" if track_confirmed else \
                f"locking({tracker.confirmed_frames}/{cfg.continuity_min_confirmed_frames})"
            refine_str = f" refining({capture_decision.refine_frames_left})" if capture_decision.refine_active else ""
            print(f"[status] guidance='{guidance_text}'  score={score:.1f}  "
                  f"sharp={sharpness:.0f}/{sharp_thresh:.0f}  vein={vein_score:.1f}/{vein_thresh:.1f}  "
                  f"adaptive={ready_str}  mode={mode_str}  track={track_str}{refine_str}  "
                  f"zoom={cfg.zoom_factor:.2f}x  exposure={exposure_ctrl.last_status}")
            # Raw camera property readback, printed every status tick (not
            # just once at startup) - the fastest way to tell whether
            # "exposure stays too high" is a software logic issue (these
            # numbers should visibly move toward the baseline over a few
            # seconds) or the camera driver just not honoring writes at all.
            print(f"  [camera] AUTO_EXPOSURE={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}  "
                  f"EXPOSURE={cap.get(cv2.CAP_PROP_EXPOSURE):.2f}  "
                  f"GAIN={cap.get(cv2.CAP_PROP_GAIN):.2f}  "
                  f"BRIGHTNESS={cap.get(cv2.CAP_PROP_BRIGHTNESS):.2f}  "
                  f"frame_mean={exposure_mean_ema:.1f}")
            last_status_print = now

        draw_overlay(display, guidance_text, score, leaf_present,
                     len(capture_decision.stability_hist), capture_decision.stability_hist.maxlen,
                     capture_decision.captures_count, now < capture_decision.flash_until)

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
            capture_decision.captures_count += 1
            _beep()
            capture_decision.flash_until = time.time() + cfg.capture_flash_sec
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
