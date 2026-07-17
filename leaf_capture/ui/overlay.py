"""On-screen overlay: guidance panel, score badge, stability bar, capture
flash, and the debug mask inset. Presentation only - no algorithmic logic."""

import cv2
import numpy as np

from quality.quality_control import Guidance


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
    """rembg blocks for roughly 0.3-1.5s - long enough that the preview
    would otherwise look frozen/hung. This draws a quick banner and
    flushes it to screen BEFORE the blocking call, so the pause reads as
    "the app is doing something" rather than "the app broke"."""
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
    accents + label) around the detected leaf."""
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
        # Hard to miss: a full-frame color wash plus a labeled banner.
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
                  covers, before any contour is picked.
      contour%  - how much of the frame the SELECTED (winning) contour
                  covers - what actually becomes the bounding box/crop.
      exposure  - the (decontaminated) brightness reading fed to the
                  auto-exposure controller this frame.
      score     - the winning candidate's raw score from
                  _score_candidates."""
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
    corner. Toggle with 'm'. White = what the detector thinks is leaf."""
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
