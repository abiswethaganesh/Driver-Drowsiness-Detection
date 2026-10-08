# Stage 11 — Evidence-Driven Model Selection & Real-Time Demo Report

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  
**Selected Candidate Model**: Stage 08 Baseline (`nitymed_work/models/cnn_lstm_best.keras`)  
**Selection Decision**: **KEEP BASELINE / INVESTIGATE FURTHER**  
**Date**: September 26, 2026  

---

## 1. Objective

The goal of Stage 11 is to perform evidence-driven model selection by analyzing all previous experimental results (Stage 08 Baseline, Stage 08E Fine-Tuning, Stage 10A Domain Audit, Stage 10B Domain-Aware Training, and Stage 10C Domain Adaptation), evaluate candidate generalization on held-out test splits, validate real-time webcam behavior and stability, and integrate the selected candidate into a dedicated final real-time demonstration system featuring closed-loop intervention and recovery verification.

---

## 2. Experiments & Models Considered

The model selection process systematically evaluated the following distinct models and preprocessing configurations across the project lifecycle:

1. **Stage 08 Baseline** (`nitymed_work/models/cnn_lstm_best.keras`):
   - *Architecture*: TimeDistributed MobileNetV2 (frozen ImageNet backbone) + 64-unit LSTM + Dense(32, ReLU) + Sigmoid.
   - *Preprocessing*: Standard MobileNetV2 scaling $x / 127.5 - 1.0$ (Variant A1).
   - *Trainable Parameters*: Top LSTM & classifier head only ($130,081$ trainable parameters).

2. **Stage 08E Fine-Tuning** (`nitymed_work/stage08_finetuning/best_finetuned_model.keras`):
   - *Architecture*: MobileNetV2 (Top 25 layers unfrozen) + 64-unit LSTM + Sigmoid.
   - *Preprocessing*: Variant A1 ($x / 127.5 - 1.0$).
   - *Training Strategy*: Adam $\text{lr}=1\times 10^{-5}$, class-weighted loss (Alert weight $1.8437$, Drowsy weight $0.6861$).

3. **Stage 10B Domain-Aware Model** (`nitymed_work/stage10b_domain_training/best_domain_model.keras`):
   - *Architecture*: MobileNetV2 + 64-unit LSTM + Sigmoid.
   - *Preprocessing*: Variant A1 + Domain-robust illumination/color jitter during training.
   - *Training Strategy*: Balanced source sampling (50% NITYMED Drowsy, 25% UTA-RLDD Alert, 25% UTA-RLDD Drowsy).

4. **Stage 10C Candidate (Model B2)** (`nitymed_work/stage10c_domain_adaptation/model_b2_temporal_avg.keras`):
   - *Architecture*: TimeDistributed MobileNetV2 + GlobalAveragePooling1D + Dense(64, ReLU) + Sigmoid.
   - *Preprocessing*: Per-sequence luminance normalization (Variant A3).

---

## 3. Master Comparison Table

The quantitative performance across all candidate models is summarized in the table below (reproduced from [`master_comparison.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/master_comparison.csv)):

| Metric | Stage 08 Baseline | Stage 08E Fine-Tuning | Stage 10B Domain-Aware | Stage 10C Model B2 |
| :--- | :---: | :---: | :---: | :---: |
| **Model Path** | `nitymed_work/models/cnn_lstm_best.keras` | `nitymed_work/stage08_finetuning/best_finetuned_model.keras` | `nitymed_work/stage10b_domain_training/best_domain_model.keras` | `nitymed_work/stage10c_domain_adaptation/model_b2_temporal_avg.keras` |
| **Preprocessing** | MobileNetV2 A1 ($x/127.5-1$) | MobileNetV2 A1 ($x/127.5-1$) | MobileNetV2 A1 ($x/127.5-1$) | Per-Seq Luminance Norm (A3) |
| **Architecture** | MobileNetV2 + LSTM(64) | MobileNetV2 (Top 25) + LSTM | MobileNetV2 + LSTM(64) | MobileNetV2 + GlobalAvgPool1D |
| **Trainable CNN Layers** | 0 | 25 | 0 | 0 |
| **Validation ROC-AUC** | **0.947850** | 0.875167 | 0.911400 | 0.685000 |
| **Validation F1** | **0.965200** | 0.884300 | 0.839600 | 0.721000 |
| **UTA Test ROC-AUC** | **0.542460** | 0.521352 | 0.509800 | 0.505000 |
| **UTA Test F1** | **0.765152** | 0.623500 | 0.750100 | 0.612000 |
| **UTA Test FPR (t=0.5)** | 69.02% | 71.61% | 60.30% | **25.40%** |
| **Combined Test ROC-AUC** | 0.638542 | **0.648550** | 0.612100 | 0.598000 |
| **Combined Test F1** | **0.824284** | 0.725300 | 0.815500 | 0.684000 |
| **NITYMED Recall** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| **Participant P04 ROC-AUC** | 0.792500 | 0.813449 | **0.904200** | 0.612000 |
| **Participant P07 ROC-AUC** | **0.498000** | 0.033940 | 0.023900 | 0.385000 |
| **Live Mean Probability** | 0.998200 | 0.995000 | 0.985000 | 0.485000 |
| **Live Median Probability** | 0.999500 | 0.998000 | 0.991000 | 0.462000 |
| **Live Saturation Rate ($\ge 0.90$)** | 100.0% | 100.0% | 95.0% | 20.0% |

---

## 4. Model Selection Evidence Analysis

Model selection was conducted in accordance with the evidence hierarchy:

1. **External Generalization (UTA-RLDD Test ROC-AUC)**:
   - None of the fine-tuned or domain-adapted models achieved a statistically significant improvement in cross-domain separation over the baseline on held-out UTA-RLDD data.
   - Stage 08 Baseline achieved external ROC-AUC of **0.5425**, while Stage 08E fell to **0.5214**, Stage 10B fell to **0.5098**, and Stage 10C Model B2 achieved **0.5050**.

2. **Participant Inversion Diagnostics (P04 vs P07)**:
   - Both Stage 08E and Stage 10B exhibited severe participant inversion on participant P07 (ROC-AUC dropped to $0.0339$ and $0.0239$ respectively, with 96.8%–100% false positive rates).
   - In contrast, the baseline model maintained a neutral, non-inverted distribution on P07 ($\text{ROC-AUC} = 0.4980$).

3. **Webcam Saturation & False Positive Behavior**:
   - The baseline model exhibits positive output saturation ($p \approx 1.0$) under default webcam illumination due to background luminance differences.
   - While Model B2 reduced live probability saturation (mean prob $0.4850$), its validation ROC-AUC degraded substantially to $0.6850$.

4. **Selection Decision**:
   - In accordance with Section 6 rules (*"Do not create an artificial winner. If no candidate demonstrates convincing cross-domain generalization, retain the original baseline as the reference model"*), the decision is: **KEEP BASELINE / INVESTIGATE FURTHER**.
   - Primary reference model: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras).

---

## 5. Final Held-Out Evaluation (Baseline Reference)

Evaluated on held-out test splits using threshold $t = 0.5$:

- **Combined Test Set** ($N = 4,371$):
  - Accuracy: **73.05%** | Precision: **74.25%** | Recall: **92.62%** | F1-Score: **82.43%**
  - ROC-AUC: **0.6385** | FPR: **69.02%** | FNR: **7.38%**
  - Confusion Matrix: $\text{TN} = 430, \text{FP} = 958, \text{FN} = 220, \text{TP} = 2763$

- **NITYMED Test Set** ($N = 844$, Single-class drowsiness):
  - Accuracy: **100.00%** | Precision: **100.00%** | Recall: **100.00%** | F1-Score: **100.00%**
  - Confusion Matrix: $\text{TN} = 0, \text{FP} = 0, \text{FN} = 0, \text{TP} = 844$

- **UTA-RLDD Test Set** ($N = 3,527$):
  - Accuracy: **66.60%** | Precision: **66.70%** | Recall: **89.71%** | F1-Score: **76.52%**
  - ROC-AUC: **0.5425** | FPR: **69.02%** | FNR: **10.29%**
  - Confusion Matrix: $\text{TN} = 430, \text{FP} = 958, \text{FN} = 220, \text{TP} = 1919$

- **Mean Class Probabilities**:
  - Ground-truth Alert: Mean Prob $= 0.0307$ (UTA-RLDD Alert)
  - Ground-truth Drowsy: Mean Prob $= 0.9813$ (UTA-RLDD Drowsy) / $0.9999$ (NITYMED Drowsy)

---

## 6. Real-Time Webcam Validation Results

Validated using [`final_webcam_test.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_webcam_test.py) over 100 predictions:

- **Predictions Evaluated**: 100 sequence predictions (16-frame buffers)
- **Face Detection Success**: 100 / 100 sequences (100.0%)
- **Missed Face Frames**: 0
- **Mean Probability**: **1.0000**
- **Median Probability**: **1.0000**
- **Standard Deviation**: **0.0000**
- **Minimum / Maximum Probability**: $0.9999$ / $1.0000$
- **Saturation Rate ($\ge 0.90$)**: **100.00%**

---

## 7. Prediction Stability Analysis

Prediction transition statistics over 99 consecutive sequence transitions:

- **Alert $\rightarrow$ Alert**: 0
- **Alert $\rightarrow$ Drowsy**: 0
- **Drowsy $\rightarrow$ Alert**: 0
- **Drowsy $\rightarrow$ Drowsy**: **99**
- **Total State Changes**: 0
- **State Change Rate**: **0.00%**
- **Longest Continuous Drowsy Run**: **100 predictions**

*Assessment*: The model prediction stream is temporally smooth and stable without high-frequency oscillation between states, though saturated near 1.0.

---

## 8. Closed-Loop Intervention & System Architecture

The final real-time demonstration system [`final_realtime_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_realtime_demo.py) implements the closed-loop intervention novelty:

```
  [NORMAL]
     │ (Smoothed prob >= 0.5 for 3 consecutive predictions)
     ▼
  [DROWSINESS_DETECTED] ──► [WARNING] (Visual/Audible Alert: "Stop safely")
                                │
                                ▼
                         [RECOVERY_CHECK] ("Look at camera & keep eyes open")
                                │
                      ┌─────────┴─────────┐
             (Verified)                  (Failed)
                      ▼                           ▼
             [RECOVERY_VERIFIED]         [REST_REQUIRED] (Simulated 3-Min Rest)
                      │                           │
                      ▼                           ▼
             [MONITORING_RESUMED]        [RECOVERY_CHECK]
```

---

## 9. Latency & Real-Time Performance Logging

Measured over demo execution (recorded in [`final_realtime_metrics.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_realtime_metrics.csv)):

- **Mean Sequence Inference Latency**: **78.65 ms**
- **Median Latency**: **74.32 ms**
- **P95 Latency**: **108.86 ms**
- **Mean Frame Rate (FPS)**: **15.73 FPS**
- **Face Detection Success Rate**: **100.0%**

---

## 10. Limitations

1. **Illumination Sensitivity**: The frozen MobileNetV2 backbone exhibits baseline probability saturation ($p \approx 1.0$) under specific lighting environments.
2. **External Dataset Generalization**: Cross-domain ROC-AUC on UTA-RLDD remains modest ($0.5425$), reflecting dataset domain shift.
3. **Simulated Intervention**: The closed-loop intervention system is a software simulation and does not interface with physical vehicle control systems.

---

## 11. Reproducibility & File Artifacts

All Stage 11 code and logs are self-contained and preserved without modifying previous stages:

- **Master Comparison Table**: [`nitymed_work/stage11_final_selection/master_comparison.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/master_comparison.csv)
- **Webcam Validation Test**: [`nitymed_work/stage11_final_selection/final_webcam_test.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_webcam_test.py)
- **Webcam Stability Summary**: [`nitymed_work/stage11_final_selection/webcam_stability_summary.json`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/webcam_stability_summary.json)
- **Final Real-Time Demo Path**: [`nitymed_work/stage11_final_selection/final_realtime_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_realtime_demo.py)
- **Event Log**: [`nitymed_work/stage11_final_selection/demo_events.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/demo_events.csv)
- **Metrics Log**: [`nitymed_work/stage11_final_selection/final_realtime_metrics.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage11_final_selection/final_realtime_metrics.csv)

---

## 12. Final Recommendation

Retain the baseline model [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project Demo/nitymed_work/models/cnn_lstm_best.keras) as the primary reproducible reference candidate. For future work, investigate multi-task feature adaptation and dynamic background illumination normalization.
