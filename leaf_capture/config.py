from dataclasses import dataclass


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

    # --- Generic fallback / hard floor. The learned adaptive target is never
    # allowed to relax below these on its own; quality_floor_relax_frac below
    # progressively lowers this floor the longer a leaf sits present without
    # a capture, so a floor that's simply uncalibrated for a given camera/
    # lens can't block every capture forever.
    sharpness_threshold: float = 90.0
    vein_score_threshold: float = 12.0
    # Fraction the hard floor above relaxes by once a leaf has been
    # continuously present for capture_timeout_sec without a capture. Ramps
    # linearly from 0 (leaf just appeared) up to this fraction. Kept small:
    # this lens is fixed-focus, so a frame that's actually out of focus
    # stays out of focus no matter how long you wait.
    quality_floor_relax_frac: float = 0.2

    # --- Stability / autocapture behaviour ---
    stability_frames_required: int = 8    # ~0.25-0.3s of "all checks pass"
    motion_threshold: float = 4.0         # fallback only, before adaptive learns the real noise floor
    capture_cooldown_sec: float = 3.0
    # Guarantee: save best-seen shot if "perfect" never hits this long.
    capture_timeout_sec: float = 8.0

    # --- Fixed exclusion zone - the bottom strip of the frame is where this
    # camera's own mount/tripod is physically visible in every frame. It's a
    # fixed region that can never contain the leaf, so it's excluded from
    # detection entirely rather than relying on color/shape gates to reject
    # it every frame. 0.0 disables this.
    ignore_bottom_frac: float = 0.22

    # --- HSV range for green leaf segmentation. Kept fairly narrow to true
    # greens: hue values below ~25 overlap skin tone, so a wide lower bound
    # causes the detector to lock onto a hand/arm instead of the leaf. This
    # is a secondary check - the primary segmentation is excess-green.
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
    # the auto-zoom logic while a leaf is being tracked; reset to
    # frame-center whenever the leaf is lost / nothing is being tracked.
    pan_x: float = 0.5
    pan_y: float = 0.5

    # --- Auto-zoom: narrow the field of view onto the leaf before relying
    # on it for anything else, so a wide/cluttered shot gets cropped away
    # and only a clean, leaf-dominated view reaches quality/capture logic.
    # Target area-ratio band to zoom toward - matches quality_score's own
    # "optimal" framing band, so "well framed by auto-zoom" and "scores
    # well" can't disagree.
    auto_zoom_target_low: float = 0.15
    auto_zoom_target_high: float = 0.35
    # How far off-center (as a fraction of frame width/height) the leaf may
    # be before auto-zoom considers it FULLY "aimed" (panning stops
    # adjusting once inside this).
    auto_zoom_center_tol: float = 0.10
    # A SEPARATE, LOOSER tolerance that only gates whether zoom is allowed
    # to run at all. apply_digital_zoom is a no-op while zoom_factor<=1.001,
    # so panning has ZERO visible effect until zoom has already taken its
    # first step - gating zoom behind the tight `centered` tolerance would
    # deadlock whenever the leaf starts more than auto_zoom_center_tol
    # off-center. This looser tolerance lets zoom begin as soon as the leaf
    # is roughly aimed at, while still refusing to zoom when it's close
    # enough to a frame edge that zooming could crop it out entirely.
    auto_zoom_safe_to_zoom_tol: float = 0.30
    auto_zoom_step: float = 0.04        # zoom_factor change per frame
    auto_pan_step: float = 0.06         # max pan-fraction shift per frame
    # If the leaf has been missing this long, give up and reset back to a
    # neutral wide view to search again.
    auto_zoom_lost_reset_sec: float = 2.0
    # If zoom has been pushing the same direction for this long without
    # area_ratio improving by at least auto_zoom_stall_improve_eps, stop
    # pushing further - guards against a runaway zoom-in that never
    # actually converges (e.g. panning lagging behind a tight crop).
    auto_zoom_stall_timeout_sec: float = 2.5
    auto_zoom_stall_improve_eps: float = 0.015

    # --- Cluttered-background robustness ---
    # Candidate contours whose w/h ratio falls outside this range are
    # rejected outright before scoring - real leaves don't come as thin
    # slivers, but background clutter edges often do.
    leaf_aspect_min: float = 0.12
    leaf_aspect_max: float = 8.0
    # Candidate scoring saturates area's contribution at this fraction of
    # the frame - a candidate at or beyond this size gets no further reward
    # for being bigger still, so a large merged background+leaf blob can't
    # out-score a smaller, more precisely-green true leaf just by size.
    leaf_size_score_saturation: float = 0.15
    # Switch hysteresis: once a track exists, a DIFFERENT candidate must
    # score at least this fraction higher before the detector switches to
    # it - stops the box hopping between two similarly-scored blobs.
    continuity_switch_margin: float = 0.2
    # Display-only smoothing of the drawn box - purely cosmetic, does not
    # affect what gets measured/saved.
    bbox_smooth_alpha: float = 0.45
    bbox_hold_frames: int = 4

    # --- Flicker ("random light flashing") smoothing ---
    # EMA-smoothing the brightness reading fed to the exposure controller
    # and gamma post-process stops one noisy/clutter-confused frame from
    # yanking either around, which otherwise reads on screen as a flash.
    brightness_ema_alpha: float = 0.25
    gamma_ema_alpha: float = 0.2
    # The brightness reading source flips between "whole frame" (no leaf)
    # and "leaf-only ROI" (leaf found) - a discontinuous jump. For the
    # first N frames after that flip, use a slower EMA alpha so the
    # reaction to the jump itself is smoothed, not just ordinary noise.
    brightness_transition_alpha: float = 0.08
    brightness_transition_hold_frames: int = 15

    # --- Cluttered-background detection tuning ---
    # A candidate only gets the temporal-continuity score boost once the
    # tracker has held the SAME object for this many consecutive frames -
    # otherwise a bad lock in the first frame or two would reinforce itself.
    continuity_min_confirmed_frames: int = 5
    # Every N frames, force one "cold" comparison with continuity disabled
    # so a better candidate elsewhere in frame can win instead of being
    # permanently out-scored by the continuity bonus.
    cold_recheck_interval_frames: int = 90

    # --- Capture refinement/rejection ---
    # Once the stability window first completes, keep evaluating for this
    # many extra frames and save the best-scoring one of them.
    capture_refine_frames: int = 3
    # If True, the timeout guarantee never saves a frame below
    # low_confidence_score - it keeps waiting instead. If False (default),
    # it saves the best-seen frame anyway, flagged "timeout"/low-confidence.
    strict_reject_blurry: bool = False

    # --- Capture-time crop refinement (rembg) ---
    # rembg is class-agnostic salient-object segmentation - it doesn't know
    # what a "leaf" is, just "the most visually prominent single foreground
    # region". It's far too slow to run every frame (~0.7-1.5 FPS), so it
    # only runs ONCE, right before a photo is saved, to refine/validate the
    # final crop - never in the live preview loop.
    use_rembg_refinement: bool = True
    rembg_model: str = "u2netp"   # lightweight variant - ~2x faster than u2net, same detection rate in testing
    rembg_mask_thresh: int = 127
    # rembg's box is only trusted over the contour method's if the region
    # it picked is itself green-dominant AND overlaps the contour method's
    # own bbox - guards against rembg confidently segmenting a completely
    # different salient object (a hand, a reflection, a shadow) that isn't
    # the leaf being tracked at all.
    rembg_color_margin_floor: float = 1.3

    # --- Live-preview tracker supervision (rembg) ---
    # rembg's capture-time refinement above only fixes the FINAL saved crop.
    # This periodically consults rembg on the LIVE frame and force-reseeds
    # the tracker if it disagrees with what's currently tracked.
    #
    # Defaulted off: live-tested and made detection noticeably worse - the
    # tracker gets force-reset every 2s, but the next few frames of ordinary
    # contour tracking (which runs without hysteresis right after a reset)
    # can drag the smoothed box back toward its own bias before the next
    # correction lands. use_rembg_refinement above is unaffected by this.
    rembg_live_supervision: bool = False
    rembg_live_recheck_interval_sec: float = 2.0

    # --- Debug ---
    show_debug_mask: bool = False
