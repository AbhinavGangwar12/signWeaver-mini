# =============================================================================
# SignWeaver | Script 1: Data Collection
# =============================================================================
# PURPOSE:
#   Captures hand landmark geometry from your webcam and saves it to a CSV
#   file (gesture_data.csv). Run this script once per gesture you want to
#   teach the model. Change GESTURE_NAME each time.
#
# USAGE:
#   1. Set GESTURE_NAME below to the word you want to record (e.g. "Hello").
#   2. Run: python 1_collect.py
#   3. Position your hand in frame, then press 'r' to start recording.
#   4. Hold your gesture steady for ~100 frames (~3-4 seconds).
#   5. Press 'q' to quit at any time.
#   6. Repeat for each new gesture (change GESTURE_NAME each time).
# =============================================================================

import cv2
import mediapipe as mp
import csv
import os
import time

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================

# ▶ CHANGE THIS for every new gesture you record.
# Examples: "Hello", "ThankYou", "Yes", "No", "ILoveYou"
GESTURE_NAME = "Hello"

# How many frames to capture per recording session
FRAMES_TO_CAPTURE = 100

# Output CSV file (rows are appended, so previous gestures are preserved)
OUTPUT_CSV = "gesture_data.csv"

# Number of hand landmarks MediaPipe provides (fixed at 21)
NUM_LANDMARKS = 21

# Each landmark has x, y, z → 21 × 3 = 63 features per frame
NUM_FEATURES = NUM_LANDMARKS * 3

# =============================================================================
# ── CSV INITIALISATION ────────────────────────────────────────────────────────
# =============================================================================

def initialise_csv(filepath: str) -> None:
    """
    Creates the CSV file with a header row if it does not already exist.
    This allows multiple runs to safely append without duplicating headers.
    """
    if not os.path.exists(filepath):
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header: label + one column per coordinate value
            header = ["label"] + [f"feat_{i}" for i in range(NUM_FEATURES)]
            writer.writerow(header)
        print(f"[INFO] Created new dataset file: {filepath}")
    else:
        print(f"[INFO] Appending to existing dataset file: {filepath}")


def append_row(filepath: str, label: str, features: list) -> None:
    """
    Appends a single data row (label + 63 coordinate floats) to the CSV.
    """
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + features)


# =============================================================================
# ── LANDMARK EXTRACTION HELPER ────────────────────────────────────────────────
# =============================================================================

def extract_landmark_features(hand_landmarks) -> list:
    """
    Given a MediaPipe NormalizedLandmarkList, flatten all 21 landmarks
    into a single list of 63 floats:  [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

    MediaPipe normalises x and y to [0, 1] relative to the frame dimensions.
    z is the depth relative to the wrist landmark (smaller = closer to camera).
    """
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


# =============================================================================
# ── MAIN COLLECTION LOOP ──────────────────────────────────────────────────────
# =============================================================================

def main():
    initialise_csv(OUTPUT_CSV)

    # ── MediaPipe Hands setup ────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,   # Video stream mode for lower latency
        max_num_hands=1,           # We only track one hand at a time
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # ── OpenCV webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam at index 0. Check your camera.")
        return

    # ── State machine variables ──────────────────────────────────────────────
    is_recording   = False   # True while actively capturing frames
    frames_saved   = 0       # Counter for how many frames have been saved
    total_saved    = 0       # Total rows written in this session (for display)

    print("\n[SignWeaver] Data Collector ready.")
    print(f"  Gesture target : '{GESTURE_NAME}'")
    print(f"  Frames to grab : {FRAMES_TO_CAPTURE}")
    print(f"  Output file    : {OUTPUT_CSV}")
    print("\n  Press  'r'  to START recording")
    print("  Press  'q'  to QUIT\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Mirror the frame so it feels like a selfie camera
        frame = cv2.flip(frame, 1)

        # ── MediaPipe processing ─────────────────────────────────────────────
        # MediaPipe requires RGB; OpenCV captures in BGR
        rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False          # Small perf optimisation
        results     = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # ── Draw hand skeleton if detected ──────────────────────────────────
        hand_detected = results.multi_hand_landmarks is not None
        if hand_detected:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

        # ── Record frames when in recording mode ────────────────────────────
        if is_recording and hand_detected:
            hand_lms = results.multi_hand_landmarks[0]
            features = extract_landmark_features(hand_lms)
            append_row(OUTPUT_CSV, GESTURE_NAME, features)
            frames_saved += 1
            total_saved  += 1

            # Stop recording once we hit the target frame count
            if frames_saved >= FRAMES_TO_CAPTURE:
                is_recording = False
                frames_saved = 0
                print(f"[INFO] Recording complete! Total rows this session: {total_saved}")

        # ── Overlay UI text on the video frame ──────────────────────────────
        # Background banner for readability
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (20, 20, 20), -1)

        # Gesture label
        cv2.putText(
            frame,
            f"Gesture: {GESTURE_NAME}",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 220, 255),   # Cyan
            1,
            cv2.LINE_AA,
        )

        # Recording status
        if is_recording:
            status_color = (0, 60, 255)   # Red while recording
            status_text  = f"● RECORDING  [{frames_saved}/{FRAMES_TO_CAPTURE}]"
        elif not hand_detected:
            status_color = (0, 165, 255)  # Orange when no hand visible
            status_text  = "No hand detected — show your hand"
        else:
            status_color = (0, 255, 100)  # Green when ready
            status_text  = "Ready — press 'r' to record"

        cv2.putText(
            frame,
            status_text,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Rows saved this session (bottom-right corner)
        cv2.putText(
            frame,
            f"Saved this session: {total_saved}",
            (frame.shape[1] - 260, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("SignWeaver — Data Collector", frame)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r") and not is_recording:
            if hand_detected:
                is_recording = True
                frames_saved = 0
                print(f"[INFO] Recording started for gesture: '{GESTURE_NAME}'")
            else:
                print("[WARN] No hand detected. Show your hand before recording.")

        elif key == ord("q"):
            print("[INFO] Quitting data collector.")
            break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session complete. Total rows saved: {total_saved}")
    print(f"[INFO] Data file: {os.path.abspath(OUTPUT_CSV)}\n")


if __name__ == "__main__":
    main()