"""Temporal bbox tracking - smooths the DRAWN box and briefly holds it
through a one- or two-frame detection dropout (common in clutter, where a
candidate contour can flicker in/out of the winning score by a hair from
one frame to the next). Display-only: guidance, quality scoring, and what
gets saved all continue to use the raw, un-smoothed per-frame detection -
this only stops the on-screen box from visibly jittering/vanishing."""


class LeafTracker:
    def __init__(self, smooth_alpha: float, hold_frames: int):
        self.smooth_alpha = smooth_alpha
        self.hold_frames = hold_frames
        self.smoothed_bbox = None
        self.frames_since_seen = 0
        # Consecutive frames the SAME track has been held (survives brief
        # hold-frame flicker, resets on a real loss). Used to decide when
        # continuity scoring is trustworthy enough to enable, and when the
        # smoothed box is stable enough to drive measurement/capture, not
        # just the display.
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
