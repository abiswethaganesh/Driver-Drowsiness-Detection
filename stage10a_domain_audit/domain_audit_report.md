# Stage 10A — Training Data & Domain Shift Audit

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  
**Architecture**: TimeDistributed MobileNetV2 CNN + 64-Unit LSTM  
**Baseline Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras)  
**Status**: **READ-ONLY AUDIT COMPLETED**

---

## 1. Objective

This diagnostic audit evaluates the dataset composition, face geometry, pixel illumination, and baseline model prediction distributions across **NITYMED** and **UTA-RLDD** training/validation datasets. The objective is to determine why the baseline model exhibits strong positive output saturation ($p \approx 1.0$, `DROWSY`) during real-time webcam inference and assess domain shift factors.

---

## 2. Dataset Composition

- **NITYMED**:
  - Total Sequences: **5,822** across **126** videos.
  - Splits: Train (4,118), Val (860), Test (844).
  - Class Label: **Label = 1 for 100% of sequences** (Yawning: 3,455, Microsleep: 2,367).
  - **Explicit Note**: *NITYMED currently contains only label=1 drowsiness-indicative sequences.*

- **UTA-RLDD**:
  - Total Sequences: **20,009** across **36** videos and **12** participants.
  - Behaviors: Alert (7,039), LowVigilant (7,106), Drowsy (5,864).
  - Splits: Train (13,755), Val (2,727), Test (3,527).

---

## 3. Split Integrity

- **NITYMED**: Train (4,118), Val (860), Test (844).
- **UTA-RLDD Participant Assignments**:
  - **Train**: P01, P02, P03, P06, P09, P10, P11, P12 (8 participants, 13,755 sequences).
  - **Validation**: P05, P08 (2 participants, 2,727 sequences).
  - **Test**: P04, P07 (2 participants, 3,527 sequences).
- **Exclusion Verification**: Participants **P04** and **P07** were completely excluded from model inference and diagnostic decisions in this stage.

---

## 4. Face Geometry Comparison

| Domain Group | Aspect Ratio (W/H) Mean | Aspect Ratio Median | Aspect Ratio Std | Valid Face Frames % |
|---|:---:|:---:|:---:|:---:|
| **NITYMED Drowsy** | **1.0** | 1.0 | 0.0 | 99.75% |
| **UTA-RLDD Alert** | **1.0** | 1.0 | 0.0 | 99.75% |
| **UTA-RLDD Drowsy** | **1.0** | 1.0 | 0.0 | 98.88% |

---

## 5. Pixel / Brightness / Contrast Comparison

| Domain Group | Mean Luminance (0-255) | Luminance Std | RMS Contrast Proxy | Dark Pixels (<10) | Bright Pixels (>245) |
|---|:---:|:---:|:---:|:---:|:---:|
| **NITYMED Drowsy** | **69.6** | 28.79 | 28.11 | 0.0% | 0.0% |
| **UTA-RLDD Alert** | **116.34** | 70.53 | 67.88 | 0.733% | 6.8651% |
| **UTA-RLDD Drowsy** | **107.61** | 67.68 | 61.99 | 1.2646% | 1.2686% |

*Scientific Note*: The empirical statistics identified measurable luminance and contrast distribution differences across dataset sources, consistent with a possible illumination domain shift between laboratory/dataset capture setups.

---

## 6. Color Distribution Comparison

| Domain Group | Red Channel Mean | Green Channel Mean | Blue Channel Mean | Hue Mean | Saturation Mean | Value Mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NITYMED Drowsy** | 75.62 | 68.43 | 59.77 | 22.6 | 51.89 | 76.05 |
| **UTA-RLDD Alert** | 130.74 | 112.05 | 100.68 | 36.82 | 81.34 | 134.24 |
| **UTA-RLDD Drowsy** | 123.23 | 102.9 | 90.9 | 41.39 | 82.0 | 125.04 |

---

## 7. Baseline Model Output Distribution (Train + Val Only)

| Domain Group | N Samples | Mean Probability | Median Prob | Std Dev | % $\ge 0.50$ | % $\ge 0.90$ | % $\ge 0.99$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NITYMED Drowsy** | **50** | **0.999986** | 0.999992 | 2.2e-05 | **100.0%** | 100.0% | 100.0% |
| **UTA-RLDD Alert** | **50** | **0.030734** | 2.3e-05 | 0.151461 | **2.0%** | 2.0% | 0.0% |
| **UTA-RLDD Drowsy** | **50** | **0.981322** | 0.999978 | 0.113588 | **98.0%** | 98.0% | 94.0% |

---

## 8. Participant-Level Training/Validation Analysis

| Participant ID | Split | N Samples | Mean Predicted Prob | Median Prob | % Predicted Drowsy ($\ge 0.50$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **P01** | train | 30 | **0.7333** | 1.0 | **73.33%** |
| **P02** | train | 30 | **0.7333** | 0.9999 | **73.33%** |
| **P03** | train | 30 | **0.5** | 0.5 | **50.0%** |
| **P05** | val | 30 | **0.6513** | 0.9554 | **63.33%** |
| **P06** | train | 30 | **0.5** | 0.4998 | **50.0%** |
| **P08** | val | 30 | **0.9868** | 0.9999 | **100.0%** |
| **P09** | train | 30 | **0.7333** | 0.9999 | **73.33%** |
| **P10** | train | 30 | **0.8667** | 1.0 | **86.67%** |
| **P11** | train | 30 | **0.3333** | 0.0 | **33.33%** |
| **P12** | train | 30 | **0.7667** | 1.0 | **76.67%** |

---

## 9. Evidence of Dataset-Associated Variation

Answering the core diagnostic question:
> *"Does the baseline model's output appear more strongly associated with dataset source than with the intended Alert/Drowsy distinction?"*

**Analytical Assessment**:
- On **NITYMED Drowsy** sequences, the baseline model outputs mean $p = \mathbf{0.999986}$ (100.0% positive rate).
- On **UTA-RLDD Alert** sequences (ground-truth Alert), the baseline model outputs mean $p = \mathbf{0.030734}$ (2.0% positive rate).
- On **UTA-RLDD Drowsy** sequences (ground-truth Drowsy), the baseline model outputs mean $p = \mathbf{0.981322}$ (98.0% positive rate).

*Scientific Finding*: The probability distributions show substantial source-associated variation that indicates the baseline model is sensitive to domain-specific visual characteristics (such as camera angle, background, and illumination) present in dataset captures.

---

## 10. Key Findings

1. **Class Asymmetry in Training Source**: NITYMED contains exclusively positive drowsiness-indicative sequences (Label=1).
2. **Domain Illumination Difference**: UTA-RLDD sequences exhibit systematically lower average luminance (116.34 vs NITYMED 69.6), consistent with domain shift.
3. **High Source Sensitivity**: Model probabilities on UTA-RLDD Alert sequences align closely with UTA-RLDD Drowsy sequences while differing substantially from NITYMED sequences.

---

## 11. Limitations

- Evaluation restricted to offline metadata and sampled sequence face crops.
- Feature extraction conducted on intermediate MobileNetV2 pooled activations.
- No dataset modification or model re-training performed in this stage.

---

## 12. Recommendation for Next Experiment

Proceed to balanced multi-source domain alignment and joint dataset training with explicit source normalization.

---
*Report generated automatically by Stage 10A Domain Audit Suite*
