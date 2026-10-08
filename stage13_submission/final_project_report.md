# Final Academic Project Report

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  
**Course / Subject**: Neural Networks & Deep Learning (NNDL)  
**Primary Baseline Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras)  
**Final System Demo**: [`nitymed_work/stage12_final_demo/final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_demo.py)  
**Date**: September 26, 2026  

---

## ABSTRACT

Driver drowsiness and microsleeps represent major hazards in commercial and industrial transportation. While conventional computer vision approaches focus solely on frame-level or sequence-level drowsiness classification, existing systems lack active intervention mechanisms to ensure driver wakefulness before resuming normal operations. This paper presents an end-to-end deep learning software prototype that pairs spatio-temporal deep neural networks with a novel closed-loop drowsiness intervention and automated recovery verification workflow. The architecture extracts spatial features using a frozen TimeDistributed MobileNetV2 CNN backbone and models temporal dynamics across 16-frame video sequences using a 64-unit Long Short-Term Memory (LSTM) network. System evaluation was conducted across two distinct benchmark sources: NITYMED (5,822 sequences) and a local Fold-1 subset of UTA-RLDD (20,009 sequences across 12 participants). The trained baseline model achieved an internal Validation ROC-AUC of 0.947850, a Combined Test ROC-AUC of 0.638542, and 100% recall on NITYMED drowsiness test sequences. External evaluation on held-out UTA-RLDD test data yielded an ROC-AUC of 0.542460 and an FPR of 69.02%, revealing significant cross-dataset illumination domain shift. Real-time inference on a standard CPU benchmark demonstrated a mean sequence latency of 78.65 ms and 15.73 FPS. The prototype integrates a state machine ($\text{NORMAL} \rightarrow \text{WARNING} \rightarrow \text{RECOVERY\_CHECK} \rightarrow \text{REST\_REQUIRED}$) that enforces interactive recovery verification ("look at camera & keep eyes open") and simulated rest protocols. We document all domain generalization limitations transparently, establishing a reproducible baseline for safety-critical driver monitoring.

---

## 1. INTRODUCTION
Industrial transportation demands continuous operator vigilance. Prolonged driving shift durations and night-time operation lead to cognitive fatigue and involuntary microsleep episodes. Computer vision systems offer a non-invasive solution for detecting facial indicators of fatigue. However, real-world deployment faces two key challenges: (1) technical domain shift across capture environments, and (2) the operational gap between issuing a passive alarm and verifying actual driver recovery.

---

## 2. PROBLEM STATEMENT
Current driver monitoring systems operate as open-loop classifiers: they output a probability score and sound an alarm upon exceeding a threshold. This design suffers from two critical flaws:
1. **Driver Habituation**: Drivers quickly ignore continuous auditory warnings without changing their physiological state.
2. **Lack of Verification**: Passive alarms provide no confirmation that the driver has regained alertness before normal vehicle operation continues.

---

## 3. OBJECTIVES
1. Develop a real-time deep learning pipeline capturing temporal facial dynamics across 16-frame sequences.
2. Combine lightweight CNN spatial representations (MobileNetV2) with recurrent temporal memory (LSTM).
3. Implement a closed-loop state machine enforcing active intervention and recovery verification.
4. Conduct rigorous multi-dataset evaluation across NITYMED and UTA-RLDD.
5. Provide honest reporting of cross-domain performance, lighting sensitivity, and system limitations.

---

## 4. RELATED APPROACH / BACKGROUND
Prior driver drowsiness detection literature falls into three primary categories:
- **Physiological Sensor Methods**: EEG, EOG, and ECG signals offer high accuracy but require intrusive physical contact `[REFERENCE TO BE ADDED]`.
- **Vehicle Telematics Methods**: Steering wheel angle and lane deviation metrics capture late-stage fatigue but fail during early cognitive onset `[REFERENCE TO BE ADDED]`.
- **Computer Vision Methods**: 2D CNNs analyze eye closure (EAR) or mouth opening (YAWN) on isolated frames, whereas 3D CNNs and CNN-LSTMs model spatio-temporal sequence patterns `[REFERENCE TO BE ADDED]`.

---

## 5. DATASETS
- **NITYMED Dataset**:
  - Official Metadata: 130 videos across 21 drivers (11 male, 10 female; Yawning: 107, Microsleep: 21).
  - Local Copy: 126 videos producing 5,822 sequences ($100\%$ positive drowsiness events, $\text{Label} = 1$). Splits: Train (4,118), Val (860), Test (844).
- **UTA-RLDD Dataset (Local Fold-1 Subset)**:
  - 36 videos across 12 participants (3 videos per participant: Alert `0`, Low Vigilant `5`, Drowsy `10`).
  - Total Sequences: 20,009 sequences (Train: 13,755, Val: 2,727, Test: 3,527). Diagnostic held-out subjects: P04 & P07.
- **Benchmark Disclaimer**:
  > *"The local UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol."*

---

## 6. DATA PREPROCESSING
Raw video streams are sampled at a frame stride of 3. Face regions are localized, cropped, and resized to $128 \times 128 \times 3$. Frames are normalized via $x / 127.5 - 1.0$ and concatenated into 16-frame sequence tensors $(1, 16, 128, 128, 3)$, spanning $530\text{ ms}$ at 30 FPS.

---

## 7. PROPOSED SYSTEM
The proposed architecture integrates deep spatio-temporal sequence classification with an interactive safety state machine:
$$\text{Video} \rightarrow \text{Face Crop} \rightarrow \text{Buffer (16)} \rightarrow \text{CNN-LSTM} \rightarrow \text{Prob} \rightarrow \text{Smoothing} \rightarrow \text{Intervention} \rightarrow \text{Recovery Check}$$

---

## 8. MODEL ARCHITECTURE
- **Spatial Backbone**: TimeDistributed MobileNetV2 (ImageNet pretrained, strictly frozen, 1280-d activation).
- **Temporal Memory**: 64-Unit LSTM modeling sequential temporal dynamics.
- **Classification Head**: Dropout(0.3) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1, Sigmoid).

---

## 9. TRAINING METHODOLOGY
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam ($\text{learning rate} = 1\times 10^{-4}$)
- **Batch Size**: 16 sequences
- **Best Model Selection**: Epoch 4 checkpoint selected based on validation loss ($0.242148$).

---

## 10. EXPERIMENTAL DESIGN
Across project stages (08, 08E, 10A, 10B, 10C, 11), multiple controlled experiments evaluated partial CNN unfreezing, domain-aware color jittering, and sequence luminance normalization. Following Stage 11 evidence analysis, the baseline model was retained for final system integration.

---

## 11. RESULTS
Empirical performance metrics across datasets:

| Dataset Split | Sample Count | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR (t=0.5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined Validation** | 3,587 | 93.76% | 95.02% | 98.06% | 0.965200 | **0.947850** | 4.85% |
| **Combined Test** | 4,371 | 73.05% | 74.25% | 92.62% | **0.824284** | **0.638542** | 69.02% |
| **NITYMED Test** | 844 | 100.00% | 100.00% | **100.00%** | 1.000000 | N/A | 0.00% |
| **UTA-RLDD Test** | 3,527 | 66.60% | 66.70% | 89.71% | **0.765152** | **0.542460** | 69.02% |

---

## 12. DOMAIN SHIFT ANALYSIS
Diagnostic audit (Stage 10A) revealed a significant pixel luminance shift between NITYMED (mean $69.60$) and UTA-RLDD Alert ($116.34$). Bright background lighting elevates frozen MobileNetV2 feature activations, causing live webcam probabilities to saturate near $1.0$. Fine-tuning top layers (Stage 08E/10B) led to severe participant-specific inversion on subject P07 ($\text{ROC-AUC} = 0.0239$).

---

## 13. REAL-TIME IMPLEMENTATION
Evaluated on native CPU benchmark:
- **Mean Inference Latency**: **78.65 ms** per 16-frame sequence
- **P95 Inference Latency**: **108.86 ms**
- **Throughput**: **15.73 FPS**
- **Automated Verification**: All 17 system verification tests passed cleanly.

---

## 14. CLOSED-LOOP INTERVENTION
When 5-window smoothed drowsiness probability exceeds $0.50$ for 3 consecutive predictions, the state machine transitions from `NORMAL` to `WARNING`, triggering visual/audible stop advice:
$$\text{⚠ DROWSINESS DETECTED — Please stop safely and take a break.}$$

---

## 15. RECOVERY VERIFICATION
Following an alert, the system enters `RECOVERY_CHECK`, prompting: *"Look at camera and keep eyes open."* It verifies face stability and eye openness over a 3-second window. Verification success transitions back to `NORMAL`, while failure escalates to `REST_REQUIRED` (simulated 3-minute rest protocol).

---

## 16. LIMITATIONS
1. Illumination domain shift between dataset capture setups.
2. External UTA-RLDD ROC-AUC ($0.5425$) near random.
3. Elevated false positive rate ($69.02\%$) on bright sequences.
4. Participant-specific feature inversion on P07.
5. Live webcam probability saturation near $1.0$.
6. NITYMED is composed exclusively of positive drowsiness events ($\text{Label} = 1$).
7. Local UTA split uses custom participant assignments rather than official 5-fold benchmark.
8. Recovery check is a software heuristic prototype.
9. Safety rest countdown is a software simulation.
10. Prototype lacks physical vehicle CAN bus integration.

---

## 17. FUTURE WORK
1. Implement dynamic local illumination normalization (CLAHE).
2. Explore domain-adversarial neural networks (DANN) for feature alignment.
3. Fuse deep sequence probabilities with explicit Eye Aspect Ratio (EAR) landmarks.
4. Deploy on automotive edge hardware (NVIDIA Jetson) with infrared camera sensors.

---

## 18. CONCLUSION
This project presents a hardened real-time driver drowsiness monitoring prototype. By combining a MobileNetV2-LSTM architecture with a closed-loop intervention and recovery verification workflow, the system moves beyond passive classification. Transparent reporting of domain shift limitations provides a defensible foundation for future safety-critical driver monitoring research.

---

## 19. REFERENCES
1. `[REFERENCE TO BE ADDED]` - Benchmark survey on vision-based driver fatigue detection.
2. `[REFERENCE TO BE ADDED]` - MobileNetV2 architecture specification and inverted residuals.
3. `[REFERENCE TO BE ADDED]` - Long Short-Term Memory networks for temporal sequence modeling.
4. `[REFERENCE TO BE ADDED]` - UTA-RLDD dataset paper and multi-stage drowsiness evaluation.

---

## 20. APPENDIX
All source code, event logs (`demo_events.csv`), metrics (`realtime_metrics.csv`), and verification logs (`final_checks.json`) are archived in `nitymed_work\stage12_final_demo\`.
