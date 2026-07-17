"""Decides WHEN to actually save a photo: progressive quality-floor
relaxation, the stability/steadiness window, the short "refine" window
that banks the best of a few extra frames, the timeout guarantee, and the
one-shot-per-presence bookkeeping. Calls into quality_control for the
actual quality metrics/guidance and into capture.save for the actual
file write."""

from collections import deque

from config import Config
from quality.quality_control import decide_guidance, quality_score
from capture.save import save_frame, refine_bbox_with_rembg, _beep
from ui.overlay import show_refining_banner


class CaptureDecision:
    def __init__(self, cfg: Config, rembg_session):
        self.cfg = cfg
        self.rembg_session = rembg_session

        self.stability_hist = deque(maxlen=cfg.stability_frames_required)
        self.last_capture_time = 0.0
        self.flash_until = 0.0
        self.captures_count = 0

        self.leaf_present_since = None
        self.captured_this_presence = False
        self.presence_best_frame = None
        self.presence_best_bbox = None
        self.presence_best_score = -1.0

        self.refine_active = False
        self.refine_frames_left = 0
        self.refine_best_frame = None
        self.refine_best_bbox = None
        self.refine_best_score = -1.0

    def update(self, adaptive, now, frame, display, window_name,
               leaf_present, leaf_found, measurement_bbox,
               area_ratio, brightness, sharpness, vein_score, motion):
        """Runs the full per-frame capture-decision pass: quality floor
        relaxation -> guidance/score -> presence bookkeeping -> stability
        window -> refine window / timeout guarantee -> save.

        Returns (guidance_text, score, sharp_thresh, vein_thresh) - the
        caller (main loop) still needs these for the status print and
        on-screen overlay."""
        cfg = self.cfg

        # The longer a leaf sits present without a capture, the more the
        # hard sharpness/vein floor eases off - keeps a floor that's
        # simply uncalibrated for this camera/lens from permanently
        # blocking every capture, without giving up quality checks
        # immediately the way the hard timeout fallback does.
        presence_elapsed = (now - self.leaf_present_since) if self.leaf_present_since is not None else 0.0
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
            if self.leaf_present_since is None:
                # leaf_present just became True, which (see LeafTracker)
                # can only happen off a genuine fresh detection, so
                # leaf_found is guaranteed True here too.
                self.leaf_present_since = now
                self.captured_this_presence = False
                self.presence_best_frame = frame.copy()
                self.presence_best_bbox = measurement_bbox
                self.presence_best_score = score
            elif leaf_found and score > self.presence_best_score:
                self.presence_best_frame = frame.copy()
                self.presence_best_bbox = measurement_bbox
                self.presence_best_score = score
        else:
            self.leaf_present_since = None

        steady_now = all_pass and motion < live_motion_threshold
        self.stability_hist.append(steady_now)
        is_stable = len(self.stability_hist) == self.stability_hist.maxlen and all(self.stability_hist)

        if self.refine_active:
            # Stability window already completed once - keep evaluating a
            # short extra stretch and bank whichever frame in it scores
            # highest, rather than committing to the exact frame that
            # happened to complete the streak first.
            if steady_now:
                if score > self.refine_best_score:
                    self.refine_best_frame = frame.copy()
                    self.refine_best_bbox = measurement_bbox
                    self.refine_best_score = score
                self.refine_frames_left -= 1
                refine_done = self.refine_frames_left <= 0
            else:
                refine_done = True  # lost stability mid-window - bank the best seen so far
            if refine_done:
                if self.rembg_session is not None:
                    show_refining_banner(display, window_name)
                final_bbox = refine_bbox_with_rembg(
                    self.refine_best_frame, self.refine_best_bbox, self.rembg_session, cfg)
                save_frame(self.refine_best_frame, cfg, self.refine_best_score, bbox=final_bbox)
                self.captures_count += 1
                _beep()
                self.last_capture_time = now
                self.flash_until = now + cfg.capture_flash_sec
                self.captured_this_presence = True
                self.stability_hist.clear()
                self.refine_active = False
        elif is_stable and (now - self.last_capture_time) > cfg.capture_cooldown_sec:
            self.refine_active = True
            self.refine_frames_left = cfg.capture_refine_frames
            self.refine_best_frame = frame.copy()
            self.refine_best_bbox = measurement_bbox
            self.refine_best_score = score
        elif (self.leaf_present_since is not None and not self.captured_this_presence
              and (now - self.leaf_present_since) > cfg.capture_timeout_sec
              and self.presence_best_frame is not None):
            low_confidence = self.presence_best_score < cfg.low_confidence_score
            if cfg.strict_reject_blurry and low_confidence:
                print("  [timeout] best frame seen is still below the quality floor - "
                      "continuing to wait instead of saving (strict_reject_blurry=True); "
                      "try moving/refocusing the leaf")
                self.leaf_present_since = now  # restart the timeout window rather than giving up
            else:
                if self.rembg_session is not None:
                    show_refining_banner(display, window_name)
                final_bbox = refine_bbox_with_rembg(
                    self.presence_best_frame, self.presence_best_bbox, self.rembg_session, cfg)
                save_frame(self.presence_best_frame, cfg, self.presence_best_score,
                           bbox=final_bbox, low_confidence=low_confidence)
                self.captures_count += 1
                _beep()
                if low_confidence:
                    print("  note: best frame seen wasn't fully sharp - try holding the leaf a "
                          "touch more still, or move it slowly to help the system find focus")
                self.last_capture_time = now
                self.flash_until = now + cfg.capture_flash_sec
                self.captured_this_presence = True
                self.stability_hist.clear()

        return guidance_text, score, sharp_thresh, vein_thresh
