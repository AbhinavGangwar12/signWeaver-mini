# =============================================================================
# SignWeaver | Script 3: Live Demo — Real-Time Sign Language to Speech
# =============================================================================
# PURPOSE:
#   Loads the trained model, opens the webcam, and translates detected hand
#   gestures into spoken words in real time using pyttsx3 (offline TTS).
#
# USAGE:
#   Ensure signweaver_model.pkl exists (run 2_train.py first), then:
#       python 3_demo.py
#
# CONTROLS:
#   'q' — quit
#   'c' — clear the spoken word history from the overlay
# =============================================================================

import cv2
import mediapipe as mp
import pickle
import threading
import time
import os
import numpy as np
import pyttsx3

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================

MODEL_PATH = "signweaver_model.pkl"

# How many consecutive frames the model must agree on before we accept the
# prediction and speak it. Higher = more stable but slightly slower response.
STABILITY_THRESHOLD = 10

# Minimum seconds between two spoken announcements of the SAME word.
# Prevents the TTS from repeating endlessly while you hold a gesture.
COOLDOWN_SECONDS = 2.5

# MediaPipe confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE  = 0.75

# Maximum words to show in the spoken history overlay
HISTORY_MAX_LENGTH = 6

# Camera fallbacks to improve reliability across Windows camera drivers
CAMERA_CANDIDATES = [
    (0, cv2.CAP_MSMF),
    (0, cv2.CAP_DSHOW),
    (0, cv2.CAP_ANY),
    (1, cv2.CAP_ANY),
]

# Treat almost-flat frames as invalid (often seen as gray feed)
MIN_FRAME_STDDEV = 2.0

# =============================================================================
# ── TEXT-TO-SPEECH HELPER ─────────────────────────────────────────────────────
# =============================================================================

def speak_in_thread(text: str) -> None:
    """
    Speaks `text` using pyttsx3 in a daemon thread so the OpenCV event loop
    is never blocked while the system is synthesising and playing audio.

    A new engine instance is created per call because pyttsx3 engines are
    NOT thread-safe and should not be shared across threads on Windows.
    This is the most reliable pattern on the Windows COM-based driver.
    """
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)    # Words per minute (default ~200)
            engine.setProperty("volume", 1.0)  # Full volume
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            # Non-fatal: TTS failure should never crash the video feed
            print(f"[TTS WARN] Speech failed: {exc}")

    t = threading.Thread(target=_speak, daemon=True)
    t.start()


# =============================================================================
# ── MODEL LOADING ─────────────────────────────────────────────────────────────
# =============================================================================

def load_model(filepath: str):
    """Deserialises and returns the pickled RandomForestClassifier."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Model file '{filepath}' not found.\n"
            "Run 2_train.py first to generate the model."
        )
    with open(filepath, "rb") as f:
        clf = pickle.load(f)
    print(f"[INFO] Model loaded from: {os.path.abspath(filepath)}")
    print(f"[INFO] Gesture classes known: {clf.classes_.tolist()}")
    return clf


# =============================================================================
# ── LANDMARK EXTRACTION (mirrors 1_collect.py exactly) ───────────────────────
# =============================================================================

def extract_landmark_features(hand_landmarks) -> list:
    """
    Flattens 21 MediaPipe landmarks into a 63-element feature vector:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

    Must be IDENTICAL to the extraction used in 1_collect.py, otherwise the
    model will receive data in a different distribution than it was trained on.
    """
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


# =============================================================================
# ── OVERLAY DRAWING HELPERS ───────────────────────────────────────────────────
# =============================================================================

def draw_rounded_rect(img, pt1, pt2, color, alpha=0.55, radius=12):
    """
    Draws a semi-transparent rounded rectangle overlay on `img`.
    Used for the HUD panels so the video is still visible beneath the UI.
    """
    overlay = img.copy()
    x1, y1  = pt1
    x2, y2  = pt2

    # Draw filled rectangle with clipped corners
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius),
                   (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius),
                   (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud(
    frame,
    current_word: str,
    stability_count: int,
    stability_threshold: int,
    history: list,
    hand_visible: bool,
    fps: float,
) -> None:
    """
    Renders all overlay elements onto `frame` in-place:
      • Top banner   — current predicted word + confidence bar
      • Bottom panel — spoken word history
      • Status icons — hand visibility, FPS
    """
    h, w = frame.shape[:2]

    # ── Top HUD banner ───────────────────────────────────────────────────────
    draw_rounded_rect(frame, (10, 10), (w - 10, 105), (15, 15, 15), alpha=0.65)

    # App title
    cv2.putText(frame, "SignWeaver", (24, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 210, 255), 1, cv2.LINE_AA)

    # Current prediction
    word_display = current_word if current_word else "—"
    cv2.putText(frame, word_display, (24, 88),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Stability progress bar (shows how close we are to confirming the gesture)
    bar_x1, bar_y1 = w - 200, 30
    bar_x2, bar_y2 = w - 25,  50
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
    if stability_threshold > 0:
        fill_ratio = min(stability_count / stability_threshold, 1.0)
        fill_x2    = int(bar_x1 + fill_ratio * (bar_x2 - bar_x1))
        bar_color  = (0, 255, 100) if fill_ratio >= 1.0 else (0, 180, 255)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), bar_color, -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (120, 120, 120), 1)
    cv2.putText(frame, "Stability", (bar_x1, bar_y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # ── History panel (bottom of frame) ─────────────────────────────────────
    if history:
        panel_h = 30 + len(history) * 26
        draw_rounded_rect(
            frame,
            (10, h - panel_h - 10),
            (250, h - 10),
            (15, 15, 15),
            alpha=0.65,
        )
        cv2.putText(frame, "Spoken:", (22, h - panel_h + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1, cv2.LINE_AA)
        for i, word in enumerate(reversed(history)):
            alpha_val = int(255 * (1.0 - i * 0.15))   # Fade older entries
            colour    = (alpha_val, alpha_val, alpha_val)
            cv2.putText(frame, word, (22, h - panel_h + 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 1, cv2.LINE_AA)

    # ── Status indicators (top-right corner) ────────────────────────────────
    hand_color = (0, 255, 80) if hand_visible else (0, 80, 255)
    hand_text  = "Hand: YES" if hand_visible else "Hand: NO "
    cv2.putText(frame, hand_text, (w - 145, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 145, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Controls reminder (top centre) ──────────────────────────────────────
    cv2.putText(frame, "q: quit   c: clear history",
                (w // 2 - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)


# =============================================================================
# ── CAMERA INITIALISATION HELPERS ────────────────────────────────────────────
# =============================================================================

def _is_valid_frame(frame) -> bool:
    """
    Returns True when `frame` looks like real image data, not an empty/flat
    gray buffer. This catches cases where the camera opens but outputs junk.
    """
    if frame is None or frame.size == 0:
        return False
    return float(np.std(frame)) >= MIN_FRAME_STDDEV


def open_camera_with_fallback():
    """
    Tries multiple (index, backend) pairs and returns the first camera that
    yields valid frames after a short warm-up.
    """
    for cam_index, backend in CAMERA_CANDIDATES:
        cap = cv2.VideoCapture(cam_index, backend)
        if not cap.isOpened():
            continue

        # Start with safe defaults; avoid forcing unsupported high resolutions.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        valid = False
        for _ in range(20):
            ret, frame = cap.read()
            if ret and _is_valid_frame(frame):
                valid = True
                break

        if valid:
            print(f"[INFO] Webcam opened (index={cam_index}, backend={backend}).")
            return cap

        cap.release()

    return None


# =============================================================================
# ── MAIN INFERENCE LOOP ───────────────────────────────────────────────────────
# =============================================================================

def main():
    print("=" * 60)
    print("  SignWeaver — Live Sign Language to Speech Demo")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    clf = load_model(MODEL_PATH)

    # ── MediaPipe Hands ──────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 1,
        min_detection_confidence = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence  = MIN_TRACKING_CONFIDENCE,
    )

    # ── Webcam ───────────────────────────────────────────────────────────────
    cap = open_camera_with_fallback()
    if cap is None:
        print("[ERROR] Could not get valid frames from webcam.")
        print("[HINT] Close other camera apps and check Windows Camera privacy settings.")
        return

    print("\n[INFO] Live demo running. Press 'q' to quit, 'c' to clear history.\n")

    # ── State variables ──────────────────────────────────────────────────────
    current_prediction  = None   # Most recent single-frame prediction
    stable_word         = None   # Last word that passed the stability check
    stability_counter   = 0      # Consecutive-frame counter for current pred
    last_spoken_word    = None   # The most recently TTS-spoken word
    last_spoken_time    = 0.0    # Epoch time of the most recent TTS call
    spoken_history      = []     # List of words spoken (shown in HUD)

    prev_frame_time     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or not _is_valid_frame(frame):
            print("[WARN] Invalid camera frame received; retrying...")
            continue

        if frame is None:
            print("[ERROR] Failed to read frame.")
            break

        # Mirror so gestures feel natural (like a selfie camera)
        frame = cv2.flip(frame, 1)

        # ── FPS calculation ──────────────────────────────────────────────────
        now             = time.time()
        fps             = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now

        # ── MediaPipe inference ──────────────────────────────────────────────
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results    = hands.process(rgb)
        rgb.flags.writeable = True

        hand_visible = results.multi_hand_landmarks is not None

        if hand_visible:
            hand_lms = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1),
            )

            # ── Feature extraction → prediction ─────────────────────────────
            features = extract_landmark_features(hand_lms)
            feature_array = np.array(features).reshape(1, -1)

            # predict() returns an array; take the first element
            raw_prediction = clf.predict(feature_array)[0]

            # ── Stabilisation logic ──────────────────────────────────────────
            # Only "lock in" a word if the same prediction holds for
            # STABILITY_THRESHOLD consecutive frames. This eliminates
            # momentary flickering between similar gestures.
            if raw_prediction == current_prediction:
                stability_counter += 1
            else:
                # New prediction detected — reset the counter
                current_prediction = raw_prediction
                stability_counter  = 1

            if stability_counter >= STABILITY_THRESHOLD:
                stable_word = current_prediction

                # ── Cooldown gate ────────────────────────────────────────────
                # Don't speak the same word again until COOLDOWN_SECONDS
                # have elapsed — prevents TTS spam while holding a gesture.
                time_since_last = now - last_spoken_time
                is_new_word     = (stable_word != last_spoken_word)
                cooldown_ok     = (time_since_last >= COOLDOWN_SECONDS)

                if is_new_word or cooldown_ok:
                    print(f"[SPEAK] '{stable_word}'")
                    speak_in_thread(stable_word)

                    # Update history
                    spoken_history.append(stable_word)
                    if len(spoken_history) > HISTORY_MAX_LENGTH:
                        spoken_history.pop(0)

                    last_spoken_word = stable_word
                    last_spoken_time = now

        else:
            # No hand in frame — reset stability counter
            # (don't reset current_prediction to avoid flicker on brief occlusion)
            stability_counter = 0

        # ── Render HUD ───────────────────────────────────────────────────────
        draw_hud(
            frame,
            current_word       = stable_word,
            stability_count    = stability_counter,
            stability_threshold= STABILITY_THRESHOLD,
            history            = spoken_history,
            hand_visible       = hand_visible,
            fps                = fps,
        )

        cv2.imshow("SignWeaver — Live Demo", frame)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting live demo.")
            break
        elif key == ord("c"):
            spoken_history.clear()
            stable_word        = None
            last_spoken_word   = None
            print("[INFO] History cleared.")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("[INFO] SignWeaver demo closed.\n")


if __name__ == "__main__":
    main()