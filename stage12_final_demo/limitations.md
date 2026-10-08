# Stage 12 — Research Limitations & System Scope

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

## Documented Research & System Limitations

The research team explicitly documents the following 12 technical, methodological, and operational limitations:

1. **Dataset Domain Shift**: Measurable illumination and background contrast shifts exist between NITYMED (darker indoor setup, mean luminance 69.60) and UTA-RLDD (brighter setup, mean luminance 116.34).
2. **External Generalization Performance**: Held-out UTA-RLDD Test ROC-AUC remains modest (**0.5425**), reflecting cross-domain evaluation challenges across camera setups.
3. **High External False Positive Rate**: Under default threshold $t = 0.5$, false positive rate on UTA-RLDD sequences is **69.02%**.
4. **Participant-Specific Inversion**: Fine-tuned and domain-aware training models exhibit extreme feature inversion on specific participants (e.g., P07 ROC-AUC drops to $0.0239$–$0.0339$).
5. **Webcam Probability Saturation**: Raw baseline model inference on live webcam feeds saturates near probability $p \approx 1.0$ due to baseline illumination sensitivity.
6. **Asymmetric Training Source**: NITYMED is composed exclusively of drowsiness-indicative sequences ($\text{Label} = 1$), acting as a recall benchmark rather than a balanced binary dataset.
7. **Custom Subset Evaluation**: The local UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol.
8. **Prototype Recovery Verification**: The eye-openness and face stability recovery check is a software heuristic prototype for workflow demonstration.
9. **Simulated Rest Protocol**: The 3-minute safety rest countdown is a software simulation and does not physically enforce driver rest.
10. **No Vehicle Control Integration**: The prototype is not connected to CAN bus, steering, braking, or physical vehicle control actuators.
11. **No Medical Diagnosis**: The system does not provide medical or clinical diagnosis of sleep disorders, narcolepsy, or physiological fatigue.
12. **No Guaranteed Safety Disclaimer**: The software prototype does not guarantee prevention of driver distraction, sleep onset, or vehicular collisions.
