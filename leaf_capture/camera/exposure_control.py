"""Camera exposure/gain/brightness control - hardware-facing, not part of
the detection/quality/capture algorithm."""

import time

import cv2
import numpy as np


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
    while gain doesn't.

    Two ways to take manual control:
      - Temporary: the i/k/o/l keys call pause_for_manual_override(), which
        pauses automatic adjustment and resumes it on its own after a
        short idle period.
      - Persistent: the "Auto Mode" slider in the Controls window calls
        enable_manual_lock()/disable_manual_lock(). While locked, this
        controller does nothing at all and the Exposure/Brightness
        sliders drive the camera directly - it will NOT auto-resume until
        the toggle is flipped back.

    Key design points:
    - The camera's own firmware auto-exposure is explicitly forced into
      manual mode at set_baseline(), since otherwise it fights this
      software loop for control of the same knob.
    - Every adjustment is clamped to a bounded distance from a known-good
      baseline recorded at startup, and a near-black/near-white frame
      triggers an immediate reset/snap rather than continued drift.
    - Not every property is actually settable on every UVC driver -
      cap.set() silently no-ops on many webcams instead of raising, so
      set_baseline() probes each property with a small test change (and
      confirms it actually changes real frames, not just the readback)
      and only budgets adjustments to properties that actually moved.
    - Step size is a fraction of each property's own configured drift
      band, so convergence speed is consistent regardless of units.
    - A small leaky-integral term is blended with the proportional error
      so a persistent (not just momentary) under/over-exposure pushes
      harder over time, damping the hunting a pure-proportional loop is
      prone to.
    - If a property stays pinned at its drift-band edge for a sustained
      stretch, that property's baseline re-centers toward the pinned
      side, freeing up headroom to keep converging.

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
        does. Restores the original value before returning either way."""
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
        write without the sensor doing anything. This applies a real,
        sizeable change and checks whether freshly-grabbed FRAMES
        actually got measurably brighter/darker."""
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
        actually honors, and records the current exposure/gain/brightness
        as the safe anchor all later adjustments are bounded around."""
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow convention: 0.25 = manual
        self._auto_exposure_manual_value = 0.25
        if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) not in (0.25,):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 convention: 1 = manual
            self._auto_exposure_manual_value = 1
        ae_readback = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"  AUTO_EXPOSURE after manual-mode request: {ae_readback}")
        # Some drivers never confirm ANY value via readback at all -
        # readback isn't just "different", it's simply not meaningful for
        # this property on this driver. Detecting that up front is what
        # stops the periodic recheck below from printing a false
        # "reverted to auto mode" warning every check when nothing has
        # actually changed.
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
        baseline if it's been pinned there for a long stretch. Returns
        (moved, applied_value) - moved is False when the requested step
        didn't actually change the reported value (already clamped to the
        nearest edge), so callers can cascade to the next lever."""
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
        isn't keeping up, so a hard, immediate correction is used
        instead."""
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
        # back to their OWN firmware auto-exposure after a while, even
        # though set_baseline() forced it to manual once at startup.
        # Re-asserting the same manual value periodically (before the
        # manual_lock/paused checks below - the sliders need this forced
        # too) is a cheap, harmless guard.
        if self._auto_exposure_manual_value is not None:
            recheck_now = time.time()
            if (recheck_now - self._last_manual_reassert) > 3.0:
                self._last_manual_reassert = recheck_now
                current_ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                # Only warn on a genuine TRANSITION (compared to the last
                # OBSERVED reading, not the originally-requested value) -
                # on drivers where readback never confirmed the manual
                # value in the first place, every check would otherwise
                # "mismatch" forever and print a false warning even though
                # nothing ever actually changed.
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

        # Safety net: a run of consecutive near-black frames means
        # whatever direction we were correcting in overshot badly - snap
        # straight back to the known-good baseline.
        if mean_brightness < 8.0:
            self._black_frame_streak += 1
            if self._black_frame_streak >= 3:
                self.reset_to_baseline(cap)
                self._black_frame_streak = 0
            return
        else:
            self._black_frame_streak = 0

        # Symmetric safety net for the opposite extreme.
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

        # Lowering gain is tried FIRST when darkening (no image-quality
        # downside, unlike raising it) - gain was previously only ever
        # raised, never lowered, which was a direct cause of "exposure
        # stays too high": gain kept amplifying the signal regardless of
        # what exposure/brightness were doing.
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
