"""
14_webcam_realtime_inference.py

Run this on the local Windows machine (needs a real display + webcam).

Pipeline:
  1. Open webcam.
  2. Detect/crop face + compute EAR/MAR every ~FRAME_STRIDE/NATIVE_FPS seconds
     (time-based sampling, so it matches the training temporal window
     regardless of the camera's actual FPS) -- SAME preprocessing as training
     (mediapipe FaceMesh crop -> resize to CFG.IMG_SIZE).
  3. Maintain a rolling buffer of the last CFG.SEQ_LEN sampled frames.
  4. First CFG.CALIBRATION_SECONDS: show "CALIBRATING", feed sequences to the
     PersonalCalibrator instead of making decisions.
  5. After calibration: run CNN+LSTM on the buffer, get personalized risk,
     display ALERT / DROWSINESS RISK, play an alarm above threshold.
  6. Shows FPS and per-sequence inference latency on-screen.

Press 'q' to quit.
"""

import os
import sys
import time
import platform
import collections
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from importlib import import_module

CFG = import_module("00_config").CFG
face_lib = import_module("06_face_landmark_extraction")
baseline_lib = import_module("11_baseline_system")
calib_lib = import_module("12_personalized_calibration")

ALARM_SUSTAIN_FRAMES = 3   # require this many consecutive high-risk decisions before alarming
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import winsound
    def play_alarm():
        winsound.Beep(1500, 400)
else:
    def play_alarm():
        print("\a", end="", flush=True)  # terminal bell fallback on non-Windows


def load_model():
    return tf.keras.models.load_model(os.path.join(CFG.MODEL_DIR, "cnn_lstm_best.keras"))


def get_face_crop_and_metrics(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lms = result.multi_face_landmarks[0].landmark
    pts_px = [(lm.x * w, lm.y * h) for lm in lms]

    ear = (face_lib.eye_aspect_ratio(pts_px, face_lib.RIGHT_EYE) +
           face_lib.eye_aspect_ratio(pts_px, face_lib.LEFT_EYE)) / 2.0
    mar = face_lib.mouth_aspect_ratio(pts_px)

    x0, y0, x1, y1 = face_lib.face_bbox_from_landmarks(pts_px, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = cv2.resize(frame[y0:y1, x0:x1], (CFG.IMG_SIZE, CFG.IMG_SIZE))
    return crop, ear, mar, (x0, y0, x1, y1)


def main():
    model = load_model()
    baseline_system = baseline_lib.BaselineSystem(model)
    calibrator = calib_lib.PersonalCalibrator(driver_id="live_session")

    sample_period = CFG.FRAME_STRIDE / CFG.NATIVE_FPS   # seconds between kept frames, matches training
    frame_buffer = collections.deque(maxlen=CFG.SEQ_LEN)
    ear_buffer = collections.deque(maxlen=CFG.SEQ_LEN)
    mar_buffer = collections.deque(maxlen=CFG.SEQ_LEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("!! could not open webcam")
        sys.exit(1)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                       refine_landmarks=True, min_detection_confidence=0.5)

    session_start = time.time()
    last_sample_time = 0.0
    calibrating = True
    high_risk_streak = 0
    last_latency_ms = 0.0
    fps_smoother = collections.deque(maxlen=30)
    prev_loop_time = time.time()

    print(f"Calibrating for {CFG.CALIBRATION_SECONDS}s -- please look at the camera normally...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        # --- FPS bookkeeping (camera loop rate, independent of sampling) ---
        loop_dt = now - prev_loop_time
        prev_loop_time = now
        if loop_dt > 0:
            fps_smoother.append(1.0 / loop_dt)
        display_fps = np.mean(fps_smoother) if fps_smoother else 0.0

        status_text = ""
        risk_text = ""

        result = get_face_crop_and_metrics(frame, face_mesh)
        if result is not None:
            crop, ear, mar, bbox = result
            x0, y0, x1, y1 = bbox
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # time-based sampling so effective rate matches training regardless of camera fps
            if now - last_sample_time >= sample_period:
                last_sample_time = now
                frame_buffer.append(crop)
                ear_buffer.append(ear)
                mar_buffer.append(mar)

            if calibrating:
                elapsed = now - session_start
                status_text = f"CALIBRATING ({elapsed:.1f}/{CFG.CALIBRATION_SECONDS}s)"
                if len(frame_buffer) == CFG.SEQ_LEN:
                    seq = np.stack(frame_buffer, axis=0)
                    t0 = time.perf_counter()
                    prob = baseline_system.predict_proba(seq[None, ...].astype(np.float32))[0]
                    last_latency_ms = (time.perf_counter() - t0) * 1000
                    calibrator.add_calibration_sample(prob, np.mean(ear_buffer), np.mean(mar_buffer))
                if elapsed >= CFG.CALIBRATION_SECONDS:
                    try:
                        calibrator.finalize_calibration()
                        calibrating = False
                        print("Calibration complete.")
                    except RuntimeError as e:
                        print(f"!! {e} -- extending calibration.")
                        session_start = now  # retry
            else:
                if len(frame_buffer) == CFG.SEQ_LEN:
                    seq = np.stack(frame_buffer, axis=0)
                    t0 = time.perf_counter()
                    raw_prob = baseline_system.predict_proba(seq[None, ...].astype(np.float32))[0]
                    last_latency_ms = (time.perf_counter() - t0) * 1000

                    p_risk, dev = calibrator.personalized_risk(raw_prob, np.mean(ear_buffer), np.mean(mar_buffer))
                    calibrator.maybe_update_baseline(p_risk, np.mean(ear_buffer), np.mean(mar_buffer))
                    decision = calibrator.decide(p_risk)

                    high_risk_streak = high_risk_streak + 1 if decision else 0
                    if high_risk_streak >= ALARM_SUSTAIN_FRAMES:
                        play_alarm()

                    status_text = "DROWSINESS RISK" if decision else "ALERT"
                    risk_text = f"risk={p_risk:.2f} raw={raw_prob:.2f} dev={dev:.2f}"
        else:
            status_text = "NO FACE DETECTED"

        color = (0, 0, 255) if status_text == "DROWSINESS RISK" else (0, 200, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if risk_text:
            cv2.putText(frame, risk_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"FPS: {display_fps:.1f}  latency: {last_latency_ms:.0f}ms",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("NITYMED Drowsiness Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
