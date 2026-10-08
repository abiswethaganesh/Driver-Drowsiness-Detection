# Stage 12 — 3-Minute Technical Presentation Script

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  

---

### [0:00 – 0:20] Problem Statement
> *"Good morning. Driver fatigue is a primary cause of catastrophic industrial transportation accidents. However, existing commercial systems stop at basic detection — issuing a simple chime that drivers easily ignore or habituate to. Detection alone is insufficient for safety-critical operations."*

---

### [0:20 – 0:45] System Architecture
> *"To address this, we present an end-to-end deep learning framework. Our pipeline captures live video, extracts facial regions into a 16-frame temporal sequence, and processes them through a TimeDistributed MobileNetV2 CNN feature extractor combined with a 64-unit LSTM temporal memory network."*

---

### [0:45 – 1:15] Real-Time Detection
> *"Our model operates in real-time, achieving a mean sequence inference latency of 78.65 ms and throughput of 15.73 FPS. To prevent false alarms caused by brief single-frame blinks, we apply a 5-prediction sliding-window temporal smoothing filter."*

---

### [1:15 – 1:50] Closed-Loop Intervention Workflow (Core Novelty)
> *"The core novelty of our project is the CLOSED-LOOP DROWSINESS INTERVENTION AND RECOVERY VERIFICATION WORKFLOW. As seen on screen, when sustained drowsiness is detected, the system transitions from NORMAL to WARNING, issuing visual and acoustic guidance advising the operator to pull over safely."*

---

### [1:50 – 2:20] Recovery Verification & Safety Rest Protocol
> *"Once stationary, the system initiates a RECOVERY CHECK, instructing the driver to look directly at the camera with eyes open. If eyes are open and face detection is stable, monitoring resumes. If verification fails, the system escalates to a simulated 3-MINUTE SAFETY REST PROTOCOL, requiring mandatory rest before re-verification."*

---

### [2:20 – 2:45] Empirical Results
> *"Empirically, our model achieves a Validation ROC-AUC of 0.947850 and 100% drowsiness recall on NITYMED test sequences. On held-out UTA-RLDD test data, our cross-domain analysis revealed an external ROC-AUC of 0.542460, highlighting critical illumination domain shift challenges between datasets."*

---

### [2:45 – 3:00] Novelty & Honest Conclusion
> *"In summary, detection alone is insufficient for safety-critical scenarios; our prototype contributes closed-loop intervention, automated recovery verification, and transparent reporting of domain shift limitations. Thank you."*
