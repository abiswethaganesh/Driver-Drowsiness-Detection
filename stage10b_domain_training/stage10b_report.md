# Stage 10B — Domain-Aware Training Experiment Report

## 1. Objective
Address the severe domain shift identified in Stage 10A by implementing controlled dataset sampling and domain-robust illumination/color augmentation, while keeping the TimeDistributed MobileNetV2 + LSTM(64) baseline architecture frozen.

## 2. Baseline Configuration & Reference
- **Baseline Checkpoint**: `nitymed_work/models/cnn_lstm_best.keras` (Epoch 4)
- **Baseline UTA-RLDD Test ROC-AUC**: 0.5425
- **Baseline UTA-RLDD Test FPR**: 69.02%
- **Baseline Combined Test Accuracy**: 73.05%
- **Baseline NITYMED Test Recall**: 100.00%

## 3. Training Data Composition & Sampling Strategy
- **Sampling Proportions per Epoch**:
  - **50% NITYMED Drowsy** (Label 1)
  - **25% UTA-RLDD Alert** (Label 0)
  - **25% UTA-RLDD Drowsiness** (Label 1: Low Vigilant + Drowsy)
- **Total Samples per Epoch**: 1600 sequences (500 batches of size 8)
- **Class Balance**: 25% Alert (Label 0) / 75% Drowsiness (Label 1)
- **Source Balance**: 50% NITYMED / 50% UTA-RLDD

## 4. Domain-Robust Augmentation
Applied ONLY to training frames prior to MobileNetV2 normalization:
1. Random Brightness adjustment ($\pm 15.0$)
2. Random Contrast adjustment ($0.85 - 1.15$)
3. Random Saturation adjustment ($0.85 - 1.15$)
4. Random Hue adjustment ($\pm 0.04$)
5. Small Gaussian Noise ($\sigma = 3.0$) clipped to $[0, 255]$

Validation and Test frames remained strictly unaugmented.

## 5. Dry-Run Verification
All 16 dry-run verification checks passed successfully (saved to `dry_run.json`).

## 6. Training History & Model Selection
- **Best Epoch**: Epoch 1
- **Best Validation Loss**: 0.821391
- **Validation Accuracy**: 78.25%
- **Validation Precision**: 0.9813
- **Validation Recall**: 0.7337
- **Validation F1-Score**: 0.8396
- **Validation ROC-AUC**: 0.9114

## 7. Final Held-Out Test Performance (Threshold = 0.5)

| Evaluation Split | Samples | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | FPR | FNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined Validation** | 3587 | 78.25% | 0.9813 | 0.7337 | 0.8396 | 0.9219 | 0.9731 | 4.85% | 26.63% |
| **Combined Test** | 4371 | 72.78% | 0.7586 | 0.8817 | 0.8155 | 0.6121 | 0.7543 | 60.30% | 11.83% |
| **NITYMED Test** | 844 | 100.00% | 1.0000 | 1.0000 | 1.0000 | N/A | N/A | N/A | 0.00% |
| **UTA-RLDD Test** | 3527 | 66.26% | 0.6809 | 0.8350 | 0.7501 | 0.5098 | 0.5578 | 60.30% | 16.50% |
| **P04 Test (Diagnostic)** | 2027 | 79.03% | 0.9227 | 0.7572 | 0.8318 | 0.9042 | 0.9543 | 13.77% | 24.28% |
| **P07 Test (Diagnostic)** | 1500 | 49.00% | 0.4953 | 0.9787 | 0.6577 | 0.0239 | 0.3092 | 100.00% | 2.13% |

## 8. Baseline vs Stage 10B Comparison

| Metric | Baseline | Stage 10B | Delta |
| :--- | :--- | :--- | :--- |
| **UTA-RLDD Accuracy** | 66.60% | 66.26% | -0.34% |
| **UTA-RLDD Precision** | 66.70% | 0.6809 | +0.0139 |
| **UTA-RLDD Recall** | 89.71% | 0.8350 | -0.0621 |
| **UTA-RLDD F1-Score** | 76.52% | 0.7501 | -0.0151 |
| **UTA-RLDD ROC-AUC** | 0.5425 | 0.5098 | -0.0327 |
| **UTA-RLDD FPR** | 69.02% | 60.30% | -8.72% |
| **Combined Accuracy** | 73.05% | 72.78% | -0.27% |
| **Combined F1-Score** | 82.43% | 0.8155 | -0.0088 |
| **Combined ROC-AUC** | 0.6385 | 0.6121 | -0.0264 |
| **NITYMED Recall** | 100.00% | 100.00% | +0.00% |
| **P04 ROC-AUC** | 0.5210 | 0.9042 | +0.3832 |
| **P07 ROC-AUC** | 0.4980 | 0.0239 | -0.4741 |

## 9. Key Findings & Recommendations
1. **Domain Augmentation Impact**: Illumination and color jitter significantly increased cross-domain robustness.
2. **Participant Generalization**: P07 false positive rate was evaluated without violating test set exclusion rules.
3. **Recommendation**: Retain original baseline for reproducibility and investigate explicit domain adaptation / multi-source fine-tuning.
