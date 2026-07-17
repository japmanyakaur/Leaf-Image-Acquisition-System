"""Rule-based leaf localization: excess-green + HSV color segmentation,
contour scoring, and switch hysteresis. No ML model is used here."""

import cv2
import numpy as np

from config import Config


def estimate_background_color(frame, margin=50):
    """Median color sampled from a ring of border patches. Last-resort
    estimate (see detect_leaf) - assumes a roughly uniform background."""
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
    """Excess Green Index: ExG = 2G - R - B, per pixel. Robust way to
    segment live green vegetation regardless of background - skin has
    R >= G so its ExG is low/negative, while a leaf's green channel
    dominates and its ExG is clearly positive.

    A plain global Otsu threshold picks whichever cutoff best separates
    the single MOST vividly-green region from everything else - if the
    real leaf is paler than some other object, Otsu's cutoff can land
    above the leaf's own ExG values and it never becomes foreground at
    all. Capping the threshold at a fixed, more inclusive ceiling keeps
    paler green objects in the mask too, at the cost of possibly
    including background as well - an acceptable trade because
    _score_candidates screens on shape/solidity/color plausibility
    rather than blindly taking the largest blob."""
    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_u8 = np.clip(exg, 0, 255).astype(np.uint8)
    exg_blur = cv2.GaussianBlur(exg_u8, (5, 5), 0)
    otsu_val, _ = cv2.threshold(exg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inclusive_thresh = min(otsu_val, 12.0)
    _, mask = cv2.threshold(exg_blur, inclusive_thresh, 255, cv2.THRESH_BINARY)
    return mask


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
    color, shape only - no notion of tracking/continuity, see
    detect_leaf's _pick_with_hysteresis for that) and returns the top_k
    best as a list of dicts (sorted best-first): {mask, bbox, area_ratio,
    score}. Returning more than one candidate is what lets detect_leaf()
    compare candidates ACROSS detector tiers (ExG vs HSV) rather than
    blindly accepting whichever tier happens to run first.

    Guards applied to every candidate before it's even scored:
      - area_ratio must fall within [min_ratio, max_ratio].
      - aspect_min/aspect_max reject thin slivers (mesh wire, table seam,
        shadow band) - real leaves don't come that elongated.
      - a candidate touching 3+ of the 4 frame edges is rejected outright -
        background, or several objects merged into one blob.
      - min_solidity rejects rough/jagged/branching blobs.
      - color margin (green minus red/blue) must be green-dominant, with
        both an absolute floor and a floor relative to the candidate's own
        brightness (raw color differences compress toward zero as
        brightness drops).
      - a per-pixel green_fraction check: at least 55% of the candidate's
        OWN pixels must individually clear a green-dominance floor, not
        just the region's mean. This is what rejects a MERGED blob (a real
        leaf fused by morphological close with adjacent weakly-green
        background) - the mean color-margin check alone can still pass a
        merged blob if the real leaf pixels drag its average past the
        floor, even though most of the blob's area isn't leaf.
      - an explicit minimum LUMA rejects genuinely dark/black objects
        directly, rather than relying on color-margin math alone (noisy
        in dark regions).
      - an explicit minimum SATURATION additionally rejects achromatic
        objects even when gamma correction's brightness scaling has
        pushed their luma up (scaling all channels equally changes luma
        but not hue/saturation).

    Scoring, for whatever survives the gates:
      - base = size_factor x (1 + color_margin factor) - size_factor
        SATURATES at size_saturation_ratio so a large blob and a
        well-framed true leaf beyond that point score identically on
        size - the blob no longer wins purely by being bigger.
      - x circularity factor (4*pi*area/perimeter^2, capped at 1.0) -
        jagged/branching clutter scores lower here even if it slipped
        past the solidity gate."""
    # A smaller, tighter close (and separate, smaller open) - a more
    # aggressive close can bridge small gaps between the true leaf and
    # adjacent background pixels that only marginally pass the color
    # threshold, fusing them into one contour before scoring ever sees
    # them as separate objects.
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
    # how many contours existed and exactly which gate rejected each one.
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
        # blob, not an isolated leaf.
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
            # A genuinely dark/black object (a cable, a clip, a shadow, a
            # tripod part) is rejected directly by how dark it is - dark,
            # low-signal regions are noisy enough that color-margin math
            # alone can average out to a small but non-zero green margin
            # by pure chance.
            rejected["luma"] += 1
            continue
        relative_margin = color_margin / max(mean_luma, 1.0)
        if color_margin <= 10.0 or relative_margin <= 0.04:
            # Raised from 4.0 based on per-candidate debug metrics: a
            # large, weakly-green background region consistently measured
            # color_margin 6-9 and won purely on size, while the actual
            # leaf consistently measured color_margin >= 10.2. 10.0 sits
            # in the clean gap between those two populations.
            rejected["color_margin"] += 1
            continue  # not actually green-dominant - never let this win on size alone

        # Per-pixel uniformity check - color_margin/relative_margin above
        # only look at the MEAN color of the whole candidate, which a
        # MERGED region (leaf fused with adjacent weakly-green background)
        # can still pass even though most of its area isn't leaf at all.
        # Requiring a high fraction of the candidate's OWN pixels to
        # individually clear a green-dominance floor directly rejects a
        # mostly-background blob regardless of how its average works out.
        region_pixels = frame[candidate_mask > 0].astype(np.float32)
        region_margin = np.minimum(region_pixels[:, 1] - region_pixels[:, 2],
                                    region_pixels[:, 1] - region_pixels[:, 0])
        green_fraction = float(np.mean(region_margin > 3.0))
        if green_fraction < 0.55:
            rejected["mixed_region"] += 1
            continue

        sat_values = hsv_frame[:, :, 1][candidate_mask > 0]
        if sat_values.size == 0 or float(np.mean(sat_values)) < 25.0:
            # Saturation is more robust than the luma floor above against
            # one failure mode: gamma correction scales all channels
            # equally, which changes luma but leaves hue/saturation
            # unchanged - so it can push a genuinely black/achromatic
            # object's luma up past the floor while its saturation
            # correctly stays low.
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
        # shows WHY a candidate won: genuinely strong on color/purity, or
        # just large (size_factor).
        for i, c in enumerate(candidates):
            print(f"      [{debug_label}] #{i} bbox={c['bbox']} area_ratio={c['area_ratio']:.4f} "
                  f"score={c['score']:.3f} color_margin={c['color_margin']:.1f} "
                  f"green_fraction={c['green_fraction']:.2f} mean_sat={c['mean_sat']:.1f} "
                  f"mean_bgr=({c['mean_bgr'][0]:.0f},{c['mean_bgr'][1]:.0f},{c['mean_bgr'][2]:.0f})")
    return candidates[:top_k]


def _pick_with_hysteresis(candidates, prev_bbox, switch_margin, min_iou=0.15):
    """Chooses which of `candidates` (already-scored, non-empty) to use
    this frame. If prev_bbox is given and some candidate substantially
    overlaps it, that candidate wins UNLESS a DIFFERENT candidate scores
    at least switch_margin higher - stops the box hopping between two
    similarly-scored blobs frame to frame. If nothing overlaps prev_bbox
    at all (or hysteresis is disabled via prev_bbox=None), the plain
    best-scoring candidate wins with no resistance."""
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
       tripod is physically visible there in every frame.

    1. Excess-green (ExG) segmentation - the primary detector. Robust to
       background AND to skin tone, since it reasons about green-channel
       dominance rather than a fixed hue window.
    2. HSV true-green hue mask - secondary check, catches cases ExG
       misses (e.g. very low-saturation lighting).

    Candidates from tiers 1 and 2 are pooled and compared TOGETHER - the
    winner is whichever candidate scores highest across BOTH tiers, not
    just whichever tier happened to run first.

    3. Background-difference - true last resort, only tried when 1 and 2
       together produce nothing at all. Assumes a roughly uniform
       background, usually false in genuinely cluttered scenes.

    prev_bbox/apply_hysteresis implement switch hysteresis (see
    _pick_with_hysteresis): once a track exists, a different candidate
    must score meaningfully higher before the detector switches to it.
    The caller is expected to pass apply_hysteresis=False until a track
    is "confirmed" (held for several consecutive frames) and periodically
    force it False again for a "cold" recheck, so an early bad lock
    can't permanently reinforce itself.

    Returns (mask_used_for_detection, bbox_or_None, area_ratio, debug_info).
    mask_used_for_detection is always returned (even on failure) so the
    caller can show it in the debug inset. debug_info is None when no
    leaf was found, otherwise a dict with "score", "raw_mask_fraction"
    (how much of the frame the raw per-pixel mask covers, before contour
    selection) and "tier" (which detector stage won)."""
    frame_area = frame.shape[0] * frame.shape[1]
    hyst_bbox = prev_bbox if apply_hysteresis else None

    # max_ratio caps how much of the frame a single candidate is allowed
    # to cover before it's rejected outright - the primary guard against
    # background merging with the leaf into one big blob and getting
    # accepted as "the leaf". Set just above cfg.max_area_ratio so a
    # legitimately large/close leaf still gets detected, but nowhere near
    # "most of the frame" can ever pass.
    max_candidate_ratio = min(cfg.max_area_ratio + 0.05, 0.60)

    # Blank out the fixed exclusion zone in EVERY mask before any contour
    # is ever extracted - the camera's own mount/tripod, physically
    # visible in the same region of every frame, can never contain the
    # leaf.
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
        # that WOULD have been foreground in the excluded strip.
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
    # so the debug inset ('m' key) can show *why* nothing matched.
    combined = cv2.bitwise_or(exg_mask, cv2.bitwise_or(hue_mask, bg_mask))
    return combined, None, 0.0, None
