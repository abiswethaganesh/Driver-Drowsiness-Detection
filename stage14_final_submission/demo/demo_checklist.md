# Final Demonstration Rehearsal Checklist

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

## 1. BEFORE DEMO CHECKLIST (Pre-Flight System Audit)

- [ ] **Laptop Power & Hardware**: Laptop charged or connected to AC power adapter.
- [ ] **Webcam Hardware**: External or integrated webcam connected, focused, and lens cleaned.
- [ ] **Camera Permissions**: Camera access permissions enabled; no competing application (Zoom, Teams, Skype) accessing video stream.
- [ ] **Python Environment**: Target virtual environment active with TensorFlow, OpenCV, Pandas, and NumPy verified.
- [ ] **Model Checkpoint**: Baseline model confirmed present at `nitymed_work\models\cnn_lstm_best.keras`.
- [ ] **Demo Launcher**: Script verified launchable via `python nitymed_work\stage12_final_demo\final_demo.py`.
- [ ] **System Checks Execution**: Automated verification script `run_final_checks.py` executed with 17/17 checks passing.
- [ ] **Demo Mode Verification**: Presentation Demo Mode tested (`python final_demo.py --demo-mode` or pressing `M` key).
- [ ] **Live Mode Verification**: Live Model Mode tested with raw inference rendering on dashboard.
- [ ] **Manual Alert Hotkey**: Pressing `E` / `e` key verified to trigger `MANUAL_ALERT` and update state to `WARNING`.
- [ ] **Recovery Verification State**: Pressing `R` / `r` key verified to initiate 3-second `RECOVERY_CHECK` stability window.
- [ ] **Rest Protocol Simulation**: Pressing `T` / `t` key verified to trigger `REST_REQUIRED` countdown ($03:00$).
- [ ] **Event Logging Verification**: Confirmed active logging to `demo_events.csv` and `realtime_metrics.csv`.

---

## 2. DURING DEMO CHECKLIST (Presentation & Rehearsal Flow)

- [ ] **0:00 – Introduction**: State problem statement (drowsiness hazard) and state that detection alone is insufficient.
- [ ] **0:20 – Live Feed Display**: Point to live camera view, face detection box, and 16-frame temporal sequence buffer.
- [ ] **0:45 – Model Performance**: Highlight mean sequence latency (78.65 ms) and throughput (15.73 FPS).
- [ ] **1:15 – Drowsiness Event Simulation**: Press `D` key to demonstrate smoothed probability elevation and `WARNING` state banner.
- [ ] **1:50 – Recovery Check Demonstration**: Press `R` key to initiate `RECOVERY_CHECK` ("Look at camera & keep eyes open").
- [ ] **2:10 – Recovery Outcomes**: Demonstrate `RECOVERY_VERIFIED` (returns to `NORMAL`) and failed recovery (`REST_REQUIRED`).
- [ ] **2:30 – Rest & Manual Alert**: Demonstrate simulated 3-minute rest protocol and press `E` key for Manual Emergency Alert.
- [ ] **2:45 – Empirical Results & Limitations**: Cite Validation ROC-AUC (0.947850), NITYMED Recall (100%), and explain UTA illumination domain shift honestly.
- [ ] **3:00 – Conclusion & Novelty**: Conclude with core novelty statement: *"Detection is only the first step; our prototype closes the loop by intervening and verifying recovery before resuming monitoring."*
