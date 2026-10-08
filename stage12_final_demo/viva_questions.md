# Stage 12 — Viva Voce Questions & Technical Defense

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

### Q1: Why did you choose MobileNetV2 as the CNN backbone?
**Answer**: MobileNetV2 uses inverted residual blocks and depthwise separable convolutions, providing a lightweight feature extractor ($2.25\text{M}$ parameters) ideal for real-time edge processing (78.65 ms latency) without overwhelming CPU resources during webcam inference.

### Q2: Why did you use an LSTM layer after the CNN?
**Answer**: Drowsiness behaviors (microsleeps, yawning duration, slow eye closure) are dynamic temporal processes. Frame-by-frame 2D CNNs miss temporal context. The 64-unit LSTM models sequential dependencies across 16 frames ($0.53\text{ seconds}$ at 30 FPS).

### Q3: Why is the sequence length set to 16 frames?
**Answer**: At 30 FPS, 16 frames span approximately $530\text{ ms}$, which cleanly captures the onset of a eye blink ($200\text{–}400\text{ ms}$) or microsleep while keeping memory consumption low enough for edge inference.

### Q4: Why resize input face crops to 128×128?
**Answer**: $128\times 128\times 3$ balances visual detail (eye/mouth landmarks remain distinguishable) with memory efficiency, reducing floating point operations (FLOPs) by $4.0\times$ compared to standard $224\times 224$ inputs.

### Q5: Why did you use the NITYMED dataset?
**Answer**: NITYMED provides high-resolution, realistic driver video sequences specifically annotated for yawning and microsleep events in automotive environments.

### Q6: Why did you incorporate the UTA-RLDD dataset?
**Answer**: UTA-RLDD provides multi-participant, multi-stage drowsiness recordings across diverse participants, enabling cross-dataset evaluation of external generalization.

### Q7: What is the primary novelty of your project?
**Answer**: The primary novelty is the **Closed-Loop Intervention & Automated Recovery Verification Workflow** ($\text{DETECT} \rightarrow \text{ASSESS} \rightarrow \text{INTERVENE} \rightarrow \text{VERIFY} \rightarrow \text{RESUME/ESCALATE}$), moving beyond passive notification.

### Q8: What is domain shift?
**Answer**: Domain shift occurs when the joint distribution of inputs and labels differs between training and test sets ($P_{train}(X, Y) \neq P_{test}(X, Y)$), such as lighting or camera differences between NITYMED and UTA-RLDD.

### Q9: Why is model performance lower on the UTA-RLDD test set (ROC-AUC = 0.5425)?
**Answer**: UTA-RLDD sequences exhibit significantly higher background luminance (116.34 vs NITYMED 69.60) and different camera angles. The frozen ImageNet CNN features extracted illumination patterns that degraded cross-domain ranking.

### Q10: Why is NITYMED Test recall 100.0%?
**Answer**: NITYMED test data consists entirely of positive drowsiness-indicative video sequences ($\text{Label} = 1$). A model tuned for high sensitivity on drowsiness features correctly identifies 100% of these positive events.

### Q11: Why is the UTA-RLDD False Positive Rate high (69.02%) under default threshold?
**Answer**: The model output distribution on UTA-RLDD Alert sequences is shifted upward due to brighter background lighting, causing many Alert frames to exceed the default $0.50$ probability threshold.

### Q12: Why does the baseline model saturate near probability 1.0 on live webcam feeds?
**Answer**: Live webcam environments match the higher brightness distribution of UTA-RLDD Alert sequences, triggering the frozen CNN backbone's sensitivity to overall frame luminance.

### Q13: Why did you retain the baseline model instead of selecting Stage 08E or Stage 10B?
**Answer**: Stage 08E and 10B caused severe participant inversion on diagnostic subject P07 ($\text{ROC-AUC} \text{ dropped to } 0.0239$) without improving held-out UTA ROC-AUC. Per Stage 11 rules, we retained the uncorrupted baseline for reproducibility.

### Q14: Why did Stage 10B domain augmentation not solve cross-domain generalization?
**Answer**: Color jitter adjusted pixel values but did not change the frozen MobileNetV2 spatial feature activation patterns in upper layers.

### Q15: Why was Stage 10C Candidate Model B2 not selected as the primary candidate?
**Answer**: While Model B2 reduced live webcam saturation, its validation ROC-AUC degraded significantly to $0.6850$ (compared to baseline $0.947850$).

### Q16: What happens after drowsiness is detected?
**Answer**: The system smooths predictions over a 5-window buffer. If sustained ($\ge 3$ consecutive predictions), it enters `WARNING`, issuing visual/audible alerts advising the driver to pull over.

### Q17: How is driver recovery verified?
**Answer**: In `RECOVERY_CHECK`, the system prompts: *"Look at camera & keep eyes open."* It evaluates face presence, eye presence, and eye openness over a 3-second stability window.

### Q18: What happens if recovery verification fails?
**Answer**: The system escalates to `REST_REQUIRED`, initiating a simulated 3-minute safety rest protocol countdown ($03:00$) before requiring re-verification.

### Q19: Does your system physically control the vehicle?
**Answer**: No. This is a research software prototype. It displays simulated guidance and does not interface with physical steering, braking, or CAN bus actuators.

### Q20: What are the main limitations of your prototype?
**Answer**: Key limitations include illumination domain shift sensitivity, high external false positive rate, raw webcam saturation, and custom non-benchmark UTA split.

### Q21: What would you improve in future work?
**Answer**: Implement dynamic adaptive histogram equalization, explicit multi-task domain adversarial training (DANN), and facial landmark Eye Aspect Ratio (EAR) fusion.

### Q22: How would you deploy this in a real vehicle?
**Answer**: Quantize the model to TensorRT/ONNX Int8 and run on an automotive edge processor (e.g., NVIDIA Jetson Orin Nano) with an infrared (IR) driver-facing camera.

### Q23: How would you validate safety before real-world deployment?
**Answer**: Conduct closed-track driving simulator trials with physiological EEG/EOG ground-truth validation under human-subjects IRB approval.

### Q24: How would you reduce false positive rates in production?
**Answer**: Combine CNN-LSTM video probabilities with explicit landmark geometry (EAR, PERCLOS, Yawn Frequency) using a multi-modal decision ensemble.

### Q25: How would you perform strict subject-independent evaluation?
**Answer**: Use Leave-One-Group-Out (LOGO) cross-validation where all sequences of a given driver are strictly excluded from training and validation splits.
