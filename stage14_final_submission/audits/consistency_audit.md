# Scientific Consistency Audit & Verification Log

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  

---

## 1. Metric Audit Matrix across Submission Artifacts

| Audited Metric / Value | Established Benchmark Value | FINAL_PPT_CONTENT.md | final_project_report.md | results_table.md | viva_defense.md | Audit Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation ROC-AUC** | **0.947850** | `0.947850` | `0.947850` | `0.947850` | `0.947850` | **MATCH** |
| **UTA Test ROC-AUC** | **0.542460** | `0.542460` | `0.542460` | `0.542460` | `0.542460` | **MATCH** |
| **UTA Test F1-Score** | **0.765152** | `0.765152` | `0.765152` | `0.765152` | `0.765152` | **MATCH** |
| **UTA Test FPR (t=0.5)** | **0.690202** (69.02%) | `69.02%` | `69.02%` | `69.02%` | `69.02%` | **MATCH** |
| **Combined Test ROC-AUC** | **0.638542** | `0.638542` | `0.638542` | `0.638542` | `0.638542` | **MATCH** |
| **Combined Test F1-Score** | **0.824284** | `0.824284` | `0.824284` | `0.824284` | `0.824284` | **MATCH** |
| **NITYMED Test Recall** | **1.000000** (100.0%) | `100.00%` | `100.00%` | `100.00%` | `100.00%` | **MATCH** |
| **Mean Sequence Latency** | **78.65 ms** | `78.65 ms` | `78.65 ms` | `78.65 ms` | `78.65 ms` | **MATCH** |
| **P95 Sequence Latency** | **108.86 ms** | `108.86 ms` | `108.86 ms` | `108.86 ms` | `108.86 ms` | **MATCH** |
| **Live Mean Probability** | **1.0000** | `1.0000` | `1.0000` | `1.0000` | `1.0000` | **MATCH** |
| **Live Probability Saturation** | **YES** | `YES` | `YES` | `YES` | `YES` | **MATCH** |

---

## 2. System Architecture & Dataset Specification Audit

- **Project Title**: *"AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation"* — **VERIFIED MATCH**
- **Model Architecture**: TimeDistributed MobileNetV2 (Frozen) + 64-Unit LSTM — **VERIFIED MATCH**
- **Baseline Model Checkpoint**: `nitymed_work\models\cnn_lstm_best.keras` — **VERIFIED MATCH**
- **Image Resolution**: $128 \times 128 \times 3$ RGB — **VERIFIED MATCH**
- **Sequence Buffer Length**: 16 frames — **VERIFIED MATCH**
- **Preprocessing Scale**: $x / 127.5 - 1.0$ (MobileNetV2 A1) — **VERIFIED MATCH**
- **NITYMED Dataset Count**: 5,822 sequences across 126 videos — **VERIFIED MATCH**
- **UTA-RLDD Dataset Count**: 20,009 sequences across 36 videos — **VERIFIED MATCH**
- **UTA Split Disclaimer**: Explicit custom Fold-1 participant split disclaimer included — **VERIFIED MATCH**

---

## 3. Consistency Audit Final Finding
> **AUDIT RESULT**: **PASS (100% Scientific & Metric Consistency Confirmed across all Stage 13 submission artifacts)**
