# Stage 12 — Final System Hardening & Research Evidence Report

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  
**Primary Baseline Model**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras)  
**Date**: September 26, 2026  

---

## 1. Final System Overview
The final system is an end-to-end deep learning software prototype designed for driver drowsiness detection and intervention in industrial transportation. It integrates real-time video processing, deep sequence modeling, and a novel closed-loop intervention state machine.

---

## 2. Final Model & Recorded Metrics
The baseline model `cnn_lstm_best.keras` was finalized for the prototype. The recorded empirical performance metrics remain uncompromised:

- **Validation ROC-AUC**: **0.947850**
- **Validation F1-Score**: **0.965200**
- **UTA-RLDD Test ROC-AUC**: **0.542460**
- **UTA-RLDD Test F1-Score**: **0.765152**
- **UTA-RLDD Test FPR (t=0.5)**: **69.02%**
- **Combined Test ROC-AUC**: **0.638542**
- **Combined Test F1-Score**: **0.824284**
- **NITYMED Test Recall**: **100.00%**
- **Live Mean Probability**: **1.0000**
- **Live Probability Saturation**: **YES**
- **Mean Inference Latency**: **78.65 ms**
- **P95 Inference Latency**: **108.86 ms**

---

## 3. Dataset Composition
- **NITYMED**: 5,822 usable 16-frame sequences across 126 videos (Train: 4,118, Val: 860, Test: 844). All sequences represent positive drowsiness events ($\text{Label} = 1$).
- **UTA-RLDD (Local Fold-1 Subset)**: 20,009 sequences across 36 videos and 12 participants (Train: 13,755, Val: 2,727, Test: 3,527). Diagnostic held-out subjects: P04 & P07.
- **Benchmark Disclaimer**: The local UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol.

---

## 4. Preprocessing Pipeline
Input video frames are cropped to face regions, resized to $128 \times 128 \times 3$, normalized using standard MobileNetV2 scaling $x / 127.5 - 1.0$, and structured into 16-frame temporal sequence tensors $(1, 16, 128, 128, 3)$.

---

## 5. Network Architecture
- **Feature Extractor**: TimeDistributed MobileNetV2 (ImageNet weights, strictly frozen, 1280-d output).
- **Temporal Memory**: 64-unit LSTM layer capturing temporal facial motion over $530\text{ ms}$.
- **Classifier Head**: Dropout(0.3) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1, Sigmoid).

---

## 6. Training Configuration
- **Optimizer**: Adam ($\text{learning rate} = 1\times 10^{-4}$)
- **Loss Function**: Binary Cross-Entropy
- **Batch Size**: 16 sequences
- **Best Checkpoint**: Selected at Epoch 4 based on validation loss ($0.242148$).

---

## 7. Evaluation & Diagnostic Findings
Evaluation across held-out test splits demonstrated strong within-domain validation performance ($\text{ROC-AUC} = 0.947850$) and 100% recall on NITYMED drowsiness sequences. On external UTA-RLDD test data, performance was impacted by domain distribution shifts.

---

## 8. Illumination Domain Shift Investigation
Diagnostic audit (Stage 10A) revealed significant pixel luminance shift between NITYMED (mean luminance $69.60$) and UTA-RLDD Alert ($116.34$). This shift causes the frozen CNN backbone output to elevate under bright lighting, resulting in elevated false positive rates on bright video sequences.

---

## 9. Real-Time Performance & Computational Efficiency
Benchmarked on native CPU inference:
- **Mean Inference Latency**: **78.65 ms** per 16-frame sequence
- **Median Inference Latency**: **74.32 ms**
- **P95 Latency**: **108.86 ms**
- **Throughput**: **15.73 FPS**

---

## 10. Closed-Loop Intervention Workflow
The primary novelty of this research is moving beyond passive detection to an active closed-loop state machine:
$$\text{NORMAL} \longrightarrow \text{WARNING} \longrightarrow \text{RECOVERY\_CHECK} \longrightarrow \text{REST\_REQUIRED}$$

---

## 11. Prototype Recovery Verification
In `RECOVERY_CHECK`, the system requires the driver to face the camera with eyes open. Upon verifying face presence and eye openness over a 3-second window, the system resumes normal monitoring (`RECOVERY_VERIFIED`). If unverified, it escalates to safety rest (`REST_REQUIRED`).

---

## 12. Presentation Demo Mode
To facilitate interactive demonstrations, [`final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_demo.py) features a dedicated **DEMO / SIMULATION MODE** accessible via the `M` key, allowing operators to trigger simulated events (`D` for drowsiness, `R` for recovery, `E` for manual emergency alert, `T` for rest countdown) without compromising live model probability reporting.

---

## 13. System Limitations
1. Illumination sensitivity of frozen CNN features.
2. External UTA-RLDD ROC-AUC ($0.5425$) near random.
3. Live webcam probability saturation ($p \approx 1.0$).
4. Absence of physical vehicle control hardware integration.

---

## 14. Recommendations for Future Work
1. Implement dynamic adaptive contrast normalization.
2. Explore domain-adversarial neural networks (DANN) to align feature space.
3. Fuse deep CNN-LSTM probabilities with geometric Eye Aspect Ratio (EAR) metrics.

---

## 15. Reproducibility & File Artifacts
All files are saved in `nitymed_work\stage12_final_demo\`:
- **Final Demo**: [`final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_demo.py)
- **Final Checks**: [`run_final_checks.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/run_final_checks.py) $\rightarrow$ [`final_checks.json`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_checks.json)
- **Research Figures**: [`figures/`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/figures/) (Figures 1–8)
- **Q&A Defense**: [`viva_questions.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/viva_questions.md)
- **Presentation Script**: [`presentation_script.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/presentation_script.md)
- **Dataset Summary**: [`dataset_summary.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/dataset_summary.md)
- **Limitations**: [`limitations.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/limitations.md)
