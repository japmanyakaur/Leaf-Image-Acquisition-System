"""
Camera Index Finder
--------------------
Cycles through camera indices 0-4, showing a live preview of each for a
few seconds so you can identify which index is your external Lenovo 300
FHD webcam vs your laptop's built-in camera.

USAGE
    python find_camera.py

Watch each window as it opens. Note the index number shown on screen when
the Lenovo feed appears. Press any key to skip to the next index early,
or just wait for the 4-second timer.

Once you know the right index, open leaf_capture_system.py and change:
    camera_index: int = 0
to the index that showed your Lenovo webcam.
"""

import cv2
import os
import time

MAX_INDEX_TO_TRY = 5
SECONDS_PER_CAMERA = 4


def main():
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    found_any = False
    for idx in range(MAX_INDEX_TO_TRY):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        found_any = True
        print(f"\nCamera index {idx} opened successfully. Showing preview for "
              f"{SECONDS_PER_CAMERA}s (press any key to skip)...")

        start = time.time()
        while time.time() - start < SECONDS_PER_CAMERA:
            ok, frame = cap.read()
            if not ok:
                break
            label = f"CAMERA INDEX = {idx}   (press any key to skip)"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.imshow("Camera Finder", frame)
            if cv2.waitKey(1) != -1:
                break

        cap.release()
        cv2.destroyAllWindows()

    if not found_any:
        print("No cameras were found on indices 0-4. Check that the Lenovo "
              "webcam is plugged in and not in use by another app.")
    else:
        print("\nDone. Use the index number that showed your Lenovo webcam "
              "as camera_index in leaf_capture_system.py's Config class.")


if __name__ == "__main__":
    main()