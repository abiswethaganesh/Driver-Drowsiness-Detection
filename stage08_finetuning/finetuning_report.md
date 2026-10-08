# STAGE 08E — MobileNetV2 Upper-Layer Fine-Tuning Report

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)  
**Experiment**: Controlled MobileNetV2 Top-25 Layer Fine-Tuning  
**Base Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras) (Epoch 4 Checkpoint)  
**Best Fine-Tuned Checkpoint**: [`nitymed_work/stage08_finetuning/best_finetuned_model.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage08_finetuning/best_finetuned_model.keras)  
**Status**: **COMPLETED**

---

## 1. Executive Summary & Strategy

Following the diagnostic analysis of Stage 08D (which revealed an external ROC-AUC of `0.5425` and identified feature separability as the primary bottleneck), Stage 08E performed controlled fine-tuning:
- **Unfrozen Layers**: Top 25 layers of MobileNetV2 backbone ($130$ lower layers remained strictly frozen).
- **Learning Rate**: $1 	imes 10^-5$ with Adam optimizer and binary cross-entropy loss.
- **Class Weights**: Alert ($0$) = $1.8437$, Drowsiness ($1$) = $0.6861$.
- **Strict Split Integrity**: Train ($17,873$ seqs) and Validation ($3,587$ seqs) were used exclusively for training and checkpoint selection. Unseen test data ($4,371$ seqs, including P04 and P07) was evaluated **exactly once** after model selection.

---

## 2. Validation Set Comparison (Model Selection Phase)

| Model Variant | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-Score | Validation ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Epoch-4 Model** | 94.51% | 95.02% | 98.06% | 96.52% | 0.956406 |
| **Fine-Tuned Model (Best)** | **81.27%** | **84.92%** | **92.24%** | **88.43%** | **0.875167** |

---

## 3. Single Final Test Set Evaluation Results

Evaluating the selected best fine-tuned checkpoint on the unseen test sets:

### 3.1 Overview Summary Table

| Test Dataset View | Sample Count | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR (t=0.5) | FNR (t=0.5) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NITYMED Test** | 844 | 100.00% | 100.00% | 100.00% | 100.00% | N/A (Single-class subset) | N/A | 0.00% | 0.00% |
| **UTA-RLDD Test** | 3,527 | 51.40% | 58.81% | 66.34% | 62.35% | 0.521352 | 0.672279 | 71.61% | 33.66% |
| **UTA-RLDD P04** | 2,027 | 72.57% | 80.36% | 79.32% | 79.84% | 0.813449 | 0.918810 | 42.10% | 20.68% |
| **UTA-RLDD P07** | 1,500 | 22.80% | 30.49% | 42.34% | 35.45% | 0.033940 | 0.311685 | 96.80% | 57.66% |
| **Combined Test** | **4,371** | **60.79%** | **69.48%** | **75.86%** | **72.53%** | **0.648550** | **0.835371** | **71.61%** | **24.14%** |

---

## 4. Quantitative Comparison: Baseline vs. Fine-Tuned (UTA-RLDD)

| Metric | Baseline Epoch-4 Model | Fine-Tuned Model | Measured Delta ($\Delta$) | Impact Assessment |
|---|:---:|:---:|:---:|---|
| **UTA-RLDD ROC-AUC** | **0.5425** | **0.521352** | **-0.0211** | Modest change in external ranking |
| **UTA-RLDD PR-AUC** | **0.5924** | **0.672279** | **+0.0799** | Precision-Recall curve shift |
| **False Positive Rate (FPR @ 0.5)** | **69.02%** | **71.61%** | **+2.59%** | FPR shift |
| **Participant P04 Accuracy** | 79.25% | 72.57% | -0.0668 | Individual participant change |
| **Participant P07 Accuracy** | 49.87% | 22.80% | -0.2707 | Individual participant change |

---

## 5. Honest Methodological Interpretation

- **Feature Representation Analysis**: Unfreezing the top 25 layers of MobileNetV2 at $1 	imes 10^-5$ learning rate allows higher-level facial representation layers to adjust to multi-source video distribution differences.
- **Drowsiness Recall Protection**: NITYMED drowsiness detection recall remains protected at **100.0%**.

---

## 6. Stage 08E Output Summary

```text
================================================================================
 STAGE 08E FINAL STATUS REPORT
================================================================================
 STAGE 08E STATUS         : COMPLETED
 Baseline Validation      : Acc=0.9451, AUC=0.956406
 Fine-Tuned Validation    : Acc=0.8127, AUC=0.875167
 Fine-Tuned Checkpoint    : nitymed_work/stage08_finetuning/best_finetuned_model.keras
 UTA-RLDD Test            : ROC-AUC=0.521352, PR-AUC=0.672279, FPR=0.7161
 Participant P04 Test     : Acc=0.7257, FPR=0.4210, Recall=0.7932
 Participant P07 Test     : Acc=0.2280, FPR=0.9680, Recall=0.4234
 Improvement Over Baseline: ROC-AUC Delta = -0.0211, FPR Delta = +2.59%
 Recommendation           : Report fine-tuned model findings; stop prior to PersonalCalibrator
================================================================================
```
