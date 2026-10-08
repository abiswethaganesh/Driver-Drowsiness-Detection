# Master Experimental Results Comparison Table

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

| Experiment / Candidate | Model Path | Preprocessing | Validation ROC-AUC | UTA Test ROC-AUC | UTA Test F1 | UTA Test FPR (t=0.5) | Combined Test ROC-AUC | Combined Test F1 | NITYMED Test Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 08 Baseline** | `nitymed_work/models/cnn_lstm_best.keras` | MobileNetV2 A1 ($x/127.5-1$) | **0.947850** | **0.542460** | **0.765152** | 69.02% | 0.638542 | **0.824284** | **100.00%** |
| **Stage 08E Fine-Tuned** | `nitymed_work/stage08_finetuning/best_finetuned_model.keras` | MobileNetV2 A1 ($x/127.5-1$) | 0.875167 | 0.521352 | 0.623500 | 71.61% | **0.648550** | 0.725300 | **100.00%** |
| **Stage 10B Domain-Aware** | `nitymed_work/stage10b_domain_training/best_domain_model.keras` | MobileNetV2 A1 ($x/127.5-1$) | 0.911400 | 0.509800 | 0.750100 | 60.30% | 0.612100 | 0.815500 | **100.00%** |
| **Stage 10C Model B2** | `nitymed_work/stage10c_domain_adaptation/model_b2_temporal_avg.keras` | Per-Seq Luminance Norm (A3) | 0.685000 | 0.505000 | 0.612000 | **25.40%** | 0.598000 | 0.684000 | **100.00%** |

---

### Diagnostic Participant Breakdown (P04 vs P07)

| Experiment / Candidate | Participant P04 Test ROC-AUC | Participant P07 Test ROC-AUC | Live Mean Probability | Live Median Probability | Live Saturation ($\ge 0.90$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 08 Baseline** | 0.792500 | **0.498000** | 0.9982 | 0.9995 | 100.0% |
| **Stage 08E Fine-Tuned** | 0.813449 | 0.033940 | 0.9950 | 0.9980 | 100.0% |
| **Stage 10B Domain-Aware** | **0.904200** | 0.023900 | 0.9850 | 0.9910 | 95.0% |
| **Stage 10C Model B2** | 0.612000 | 0.385000 | 0.4850 | 0.4620 | 20.0% |
