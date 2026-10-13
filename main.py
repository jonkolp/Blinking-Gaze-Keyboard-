"""
Main application entry point for the Gaze Keyboard assistive typing system.

Coordinates video capture, MediaPipe FaceMesh processing, head pose and gaze
estimation, Eye Aspect Ratio (EAR) blink detection, and virtual keyboard rendering.
"""

import sys
import time
import cv2
import mediapipe as mp
import numpy as np

from gaze import estimate_gaze, GazeSmoother
from helpers import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, eye_aspect_ratio
from keyboard import VirtualKeyboard


# ============================================================================
# Configuration Parameters
# ============================================================================

CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20          # Eye Aspect Ratio threshold for blink detection
BLINK_INTERVAL = 1.8          # Debounce delay (seconds) to prevent double-strikes
SMOOTHING_ALPHA = 0.35        # EMA factor for gaze smoothing (0.0=steady, 1.0=raw)


def draw_hud(frame: np.ndarray, ear: float, is_blinking: bool,
             selected_key: str, last_typed: str, cooldown_ratio: float):
    """Render informative Heads-Up Display (HUD) overlay onto the webcam feed."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Blink / EAR status
    status_text = "BLINK!" if is_blinking else "OPEN"
    status_color = (0, 0, 255) if is_blinking else (0, 255, 0)
    cv2.putText(frame, f"EAR: {ear:.3f} [{status_text}]", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Targeted Key
    target_str = f"Target: {selected_key if selected_key else 'None'}"
    cv2.putText(frame, target_str, (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Last Typed Character
    typed_str = f"Last Pressed: {last_typed if last_typed else 'None'}"
    cv2.putText(frame, typed_str, (240, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    # Debounce cooldown bar
    bar_x, bar_y, bar_w, bar_h = w - 160, 25, 140, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * min(1.0, max(0.0, cooldown_ratio)))
    bar_color = (0, 255, 0) if cooldown_ratio >= 1.0 else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, "Blink Ready" if cooldown_ratio >= 1.0 else "Cooldown",
                (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    # Bottom exit guide
    cv2.putText(frame, "Press ESC or 'q' to exit", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    """Main execution loop for gaze-based virtual keyboard typing."""
    print("=" * 60)
    print(" Gaze Keyboard System - Initializing")
    print("=" * 60)
    print(f" Camera index:    {CAMERA_INDEX}")
    print(f" EAR threshold:   {EAR_THRESHOLD}")
    print(f" Blink debounce:  {BLINK_INTERVAL}s")
    print("=" * 60)

    # Initialize video capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam at index {CAMERA_INDEX}.", file=sys.stderr)
        print("Please ensure your camera is connected and not in use by another app.", file=sys.stderr)
        sys.exit(1)

    # Initialize modules
    keyboard = VirtualKeyboard()
    gaze_smoother = GazeSmoother(alpha=SMOOTHING_ALPHA)
    mp_face_mesh = mp.solutions.face_mesh

    blink_timer = 0.0
    last_typed_key = ""

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                continue

            current_time = time.time()
            cooldown_ratio = (current_time - blink_timer) / BLINK_INTERVAL

            # Convert to RGB for MediaPipe inference
            image.flags.writeable = False
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            image.flags.writeable = True

            selected_key = None
            is_blinking = False
            avg_ear = 0.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                # 1. Gaze Estimation & Filtering
                raw_gaze = estimate_gaze(image, landmarks)
                smoothed_gaze = gaze_smoother.update(raw_gaze)

                # 2. Blink Detection (Soukupova & Cech EAR)
                left_ear = eye_aspect_ratio(
                    [landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS],
                    image.shape
                )
                right_ear = eye_aspect_ratio(
                    [landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS],
                    image.shape
                )
                avg_ear = (left_ear + right_ear) / 2.0
                is_blinking = (left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD)

                # 3. Virtual Keyboard Key Hit-Testing
                if smoothed_gaze is not None:
                    selected_key = keyboard.detect_key(smoothed_gaze)

                # 4. Keystroke Confirmation via Blink
                if is_blinking and selected_key:
                    if cooldown_ratio >= 1.0:
                        keyboard.press_key(selected_key)
                        last_typed_key = selected_key
                        blink_timer = current_time
                        print(f"[TYPED] '{selected_key}' -> Current text: '{keyboard.text}'")
            else:
                gaze_smoother.reset()
                keyboard.selected_key = None

            # Render Heads-Up Display onto webcam feed
            draw_hud(image, avg_ear, is_blinking, selected_key, last_typed_key, cooldown_ratio)

            # Render Virtual Keyboard canvas
            keyboard_canvas = keyboard.render()

            # Display windows
            cv2.imshow("Gaze Tracking Feed", image)
            cv2.imshow("Virtual Keyboard", keyboard_canvas)

            # Handle exit key
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("\n[INFO] Exiting Gaze Keyboard.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
