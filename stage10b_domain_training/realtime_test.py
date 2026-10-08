"""
STAGE 10B — STANDALONE REAL-TIME DIAGNOSTIC SCRIPT
Tests best_domain_model.keras on live webcam stream without altering realtime_webcam.py.
"""
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_domain_model.keras")
print(f"Loading Stage 10B Model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Warmup
dummy = np.zeros((1, 16, 128, 128, 3), dtype=np.float32)
for _ in range(3):
    model.predict(dummy, verbose=0)
print("Warmup complete. Starting webcam stream...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam!")
    exit(1)

buffer = deque(maxlen=16)
smooth_buffer = deque(maxlen=5)

print("\nPress 'q' to exit. Observe predictions for Alert vs Drowsy state.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize & normalize frame to [-1, 1]
    img = cv2.resize(frame, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
    
    buffer.append(img_norm)
    
    raw_prob = 0.0
    status = "BUFFERING..."
    color = (255, 255, 0)
    
    if len(buffer) == 16:
        seq_tensor = np.expand_dims(np.array(buffer), axis=0) # (1, 16, 128, 128, 3)
        raw_prob = float(model.predict(seq_tensor, verbose=0)[0][0])
        smooth_buffer.append(raw_prob)
        smoothed_prob = float(np.mean(smooth_buffer))
        
        if smoothed_prob >= 0.5:
            status = f"DROWSY ({smoothed_prob:.1%})"
            color = (0, 0, 255)
        else:
            status = f"ALERT ({smoothed_prob:.1%})"
            color = (0, 255, 0)
            
    cv2.putText(frame, f"Stage 10B Realtime Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Raw Prob: {raw_prob:.4f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.imshow("Stage 10B Realtime Diagnostic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
