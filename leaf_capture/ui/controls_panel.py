"""Controls side panel - sliders for Exposure / Brightness / Zoom plus an
Auto/Manual toggle. This is a second OpenCV window docked next to the
preview (OpenCV trackbars must live in their own window), positioned so
it reads as a side panel rather than a separate, unrelated window."""

import cv2

from config import Config
from camera.exposure_control import AutoExposureController

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
    controller owns those properties and the sliders are inert."""
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
