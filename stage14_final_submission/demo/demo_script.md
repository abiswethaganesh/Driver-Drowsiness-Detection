# 3-Minute Live Demonstration Rehearsal Script

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System  
**Demo Script Execution File**: [`nitymed_work/stage12_final_demo/final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_demo.py)  

---

### SECTION 1 — INTRODUCTION (20 Seconds)
- **Action**: Launch application in interactive mode (`python nitymed_work\stage12_final_demo\final_demo.py`).
- **Speaker**:
  > *"Good morning committee. Commercial driver fatigue causes devastating industrial accidents. Current commercial solutions merely sound an alarm when drowsiness is detected. However, detection alone does not ensure driver safety. Our research prototype introduces a Closed-Loop Intervention and Recovery Verification system."*

---

### SECTION 2 — LIVE MONITORING (40 Seconds)
- **Action**: Point out live dashboard features on the OpenCV screen.
- **Speaker**:
  > *"As shown on screen, the system captures live webcam frames, localizes the driver's face, and fills a 16-frame sequence buffer. The TimeDistributed MobileNetV2 and LSTM model evaluates temporal dynamic changes, updating the drowsiness probability meter. The system maintains smooth real-time execution with a mean sequence inference latency of 78.65 ms and 15.73 FPS."*

---

### SECTION 3 — DROWSINESS EVENT (30 Seconds)
- **Action**: Press the `D` key to activate demo mode drowsiness simulation.
- **Speaker**:
  > *"We now simulate a sustained drowsiness event. As the 5-prediction sliding-window probability exceeds 50% for consecutive frames, the state machine transitions from NORMAL to WARNING. The system presents a high-visibility visual banner advising the driver to safely stop the vehicle."*

---

### SECTION 4 — RECOVERY VERIFICATION (40 Seconds)
- **Action**: Press `R` key to trigger RECOVERY CHECK.
- **Speaker**:
  > *"Following the warning, the system initiates a RECOVERY CHECK, requiring the driver to look directly at the camera with eyes open. Upon verifying face presence and eye openness over a 3-second stability window, the system confirms RECOVERY VERIFIED and automatically resumes normal safety monitoring."*

---

### SECTION 5 — REST ESCALATION & MANUAL ALERT (30 Seconds)
- **Action**: Press `T` key to demonstrate REST REQUIRED, then press `E` key for Manual Emergency Alert.
- **Speaker**:
  > *"If recovery verification fails, the system escalates to SAFETY REST REQUIRED, enforcing a simulated 3-minute safety rest protocol countdown (03:00). Drivers or fleet managers can also manually trigger an emergency override at any time via the MANUAL ALERT hotkey, which is logged to our event database."*

---

### SECTION 6 — RESULTS & DOMAIN SHIFT (20 Seconds)
- **Action**: Direct attention to presentation slide / results summary.
- **Speaker**:
  > *"Empirically, our model achieved a Validation ROC-AUC of 0.947850 and 100% drowsiness recall on NITYMED test data. External held-out testing on UTA-RLDD yielded an ROC-AUC of 0.542460, highlighting critical illumination domain shift between dark training setups and bright test environments."*

---

### SECTION 7 — CORE NOVELTY & CONCLUSION (20 Seconds)
- **Action**: Press `Q` key to cleanly close demo and print performance summary log.
- **Speaker**:
  > *"In conclusion, detection is only the first step. Our prototype closes the loop by intervening and verifying driver recovery before resuming monitoring. Thank you, and we welcome your questions."*
