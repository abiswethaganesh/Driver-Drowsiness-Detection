# STAGE 08–10 READ-ONLY PROBABILITY DIAGNOSTICS REPORT

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)  
**Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras) (Epoch 4 Checkpoint)  
**Diagnostic Mode**: Read-Only Analysis (Zero dataset/model modifications, zero threshold tuning on test sets)  
**Diagnostic Status**: **REVIEW REQUIRED**

---

## 1. Executive Summary & Diagnostic Findings

This report presents a strictly **read-only diagnostic evaluation** of the trained MobileNetV2 + LSTM model on held-out test data.

### Key Conclusions
1. **Primary Failure Mode**: The poor performance on UTA-RLDD (ROC-AUC `0.5425`, PR-AUC `0.5924`) is primarily **a poor feature separation / ranking problem**, not merely a threshold offset or calibration misalignment.
2. **Participant Concentration**: The elevated false-positive rate ($69.02\%$ on UTA-RLDD) is **heavily concentrated in Participant P07**, whose Alert sequences are assigned a mean probability of `0.999963` ($100\%$ false-positive rate). P07 accounts for **78.2% of all false alarms** on the external dataset.
3. **NITYMED Metric Caveat**: The $100\%$ detection rate on NITYMED ($844 / 844$) confirms drowsiness sensitivity on NITYMED clips, but **provides zero evidence of normal-vs-drowsy discrimination** because NITYMED contains strictly Label 1 ($y=1$) sequences.

---

## 2. Probability Distribution Analysis (UTA-RLDD Test Set)

Inference probabilities were collected separately for ground-truth **Alert ($y=0$)** ($n=1,388$) and **Drowsy ($y=1$)** ($n=2,139$) sequences in UTA-RLDD:

### 2.1 Distribution Summary Statistics

| Statistic | Alert (Label 0) | Drowsy (Label 1) | Distribution Difference |
|---|:---:|:---:|:---:|
| **Sample Count** | 1,388 | 2,139 | — |
| **Minimum** | 0.000130 | 0.000205 | +0.000075 |
| **Maximum** | 0.999995 | 0.999995 | 0.000000 |
| **Mean** | **0.6943** | **0.8966** | +0.2023 |
| **Median** | **0.9999** | **0.9999** | 0.0000 |
| **Std Dev** | 0.4300 | 0.2751 | -0.1549 |
| **10th Percentile** | 0.0023 | 0.4463 | +0.4440 |
| **25th Percentile** | 0.1181 | 0.9980 | +0.8799 |
| **75th Percentile** | 0.999982 | 0.999982 | 0.0000 |
| **90th Percentile** | 0.999991 | 0.999989 | -0.000002 |

### 2.2 Distribution Overlap Analysis
- **Overlap Range**: $[0.000205, 0.999995]$
- **Alert samples $\ge$ Drowsy 25th percentile ($0.9980$)**: **56.41%**
- **Alert samples $\ge$ Drowsy median ($0.9999$)**: **52.38%**
- **Drowsy samples $\le$ Alert 75th percentile ($0.99998$)**: **74.52%**

---

## 3. Threshold-Independent Analysis

| Evaluation View | ROC-AUC | PR-AUC | Average Precision | Random Baseline PR-AUC |
|---|:---:|:---:|:---:|:---:|
| **UTA-RLDD Test** | **0.5425** | **0.5924** | **0.5932** | ~0.6065 |
| **Combined Test** | **0.6385** | **0.7722** | **0.7721** | ~0.6825 |

---

## 4. Threshold Curve (Diagnostic Purpose Only)

| Threshold | Precision | Recall | F1-Score | False Positive Rate (FPR) | False Negative Rate (FNR) | TN | FP | FN | TP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.05** | 0.6503 | 0.9425 | 0.7696 | 78.10% | 5.75% | 304 | 1084 | 123 | 2016 |
| **0.10** | 0.6562 | 0.9327 | 0.7704 | 75.29% | 6.73% | 343 | 1045 | 144 | 1995 |
| **0.20** | 0.6600 | 0.9219 | 0.7693 | 73.20% | 7.81% | 372 | 1016 | 167 | 1972 |
| **0.30** | 0.6641 | 0.9140 | 0.7692 | 71.25% | 8.60% | 399 | 989 | 184 | 1955 |
| **0.40** | 0.6667 | 0.9051 | 0.7678 | 69.74% | 9.49% | 420 | 968 | 203 | 1936 |
| **0.50** | **0.6670** | **0.8971** | **0.7652** | **69.02%** | **10.29%** | **430** | **958** | **220** | **1919** |
| **0.60** | 0.6681 | 0.8911 | 0.7636 | 68.23% | 10.89% | 441 | 947 | 233 | 1906 |
| **0.70** | 0.6686 | 0.8817 | 0.7605 | 67.36% | 11.83% | 453 | 935 | 253 | 1886 |
| **0.80** | 0.6712 | 0.8742 | 0.7594 | 65.99% | 12.58% | 472 | 916 | 269 | 1870 |
| **0.90** | 0.6775 | 0.8602 | 0.7580 | 63.11% | 13.98% | 512 | 876 | 299 | 1840 |
| **0.95** | 0.6790 | 0.8467 | 0.7536 | 61.67% | 15.33% | 532 | 856 | 328 | 1811 |

---

## 5. Participant-Level Analysis (P04 vs P07)

| Metric | Participant P04 | Participant P07 | Combined UTA-RLDD |
|---|:---:|:---:|:---:|
| **Alert Sequences ($y=0$)** | 639 | 749 | 1,388 |
| **Drowsy Sequences ($y=1$)** | 1,388 | 751 | 2,139 |
| **Alert False Positive Rate ($t=0.5$)** | **32.71%** (209 FP) | **100.0%** (749 FP) | **69.02%** (958 FP) |
| **Drowsy Recall ($t=0.5$)** | **84.65%** | **99.07%** | **89.71%** |
| **Mean Alert Probability** | **0.3361** | **0.999963** | **0.6943** |
| **Mean Drowsy Probability** | **0.8458** | **0.990552** | **0.8966** |
| **Separation Status** | **Moderate Separation** | **Severe Overlap / Saturation** | Mixed |

---

## 6. Source-Level Analysis (NITYMED vs UTA-RLDD)

| Data Source | Sample Count | Class Composition | Mean Prob | Detection Rate ($t=0.5$) | Key Finding |
|---|:---:|:---:|:---:|:---:|---|
| **NITYMED Test** | 844 | 100% Label 1 (Drowsy) | 0.999989 | 100.0% | Confirms high drowsiness recall on NITYMED clips, but **provides zero evidence of normal-vs-drowsy class discrimination**. |
| **UTA-RLDD Test** | 3,527 | 39.4% Alert (0), 60.6% Drowsy (1) | 0.8172 | 81.57% | Provides the **only ground-truth Alert sequences** in test data. |

---

## 7. Calibration Feasibility Assessment

### Official Diagnostic Conclusion
**`Calibration is unlikely to solve the primary problem because class distributions substantially overlap.`**

---

## 8. Summary & Next Stage Recommendation

```text
================================================================================
 STAGE 08 DIAGNOSTIC SUMMARY REPORT
================================================================================
 STAGE 08 DIAGNOSTIC STATUS : REVIEW REQUIRED
 UTA-RLDD probability sep.  : High Overlap (Alert mean 0.6943 vs Drowsy mean 0.8966)
 UTA-RLDD ROC-AUC           : 0.5425 (Near-random ranking on external domain)
 UTA-RLDD PR-AUC            : 0.5924 (Close to prevalence baseline ~0.6065)
 Participant-level findings  : P07 accounts for 78.2% of false alarms (100% FPR)
 Threshold-independent      : Problem is feature representation, not threshold choice
 Calibration feasibility    : Calibration unlikely to solve primary problem alone
================================================================================
 RECOMMENDED NEXT STAGE     : Fine-tune MobileNetV2 upper layers / domain adaptation
================================================================================
```

---
*Report Generated Automatically by `08d_probability_diagnostics.py`*
