# Stage 12 — Dataset Composition & Split Integrity Summary

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  
**Datasets Used**: NITYMED + UTA-RLDD  

---

## 1. NITYMED Dataset Composition

### Official Dataset Metadata
- **Total Official Videos**: 130 videos
- **Total Participant Count**: 21 drivers (11 male, 10 female)
- **Official Behavior Annotations**:
  - Yawning: 107 videos
  - Microsleep: 21 videos

### Local Usable Dataset Processing
- **Total Processed Videos**: 126 videos (19 Microsleep, 107 Yawning)
- **Total Generated 16-Frame Sequences**: **5,822 sequences**
- **Data Splits**:
  - **Train Set**: 4,118 sequences
  - **Validation Set**: 860 sequences
  - **Test Set**: 844 sequences
- **Class Label Property**: 100% of NITYMED sequences represent positive drowsiness-indicative events ($\text{Label} = 1$). NITYMED serves as a positive drowsiness recall benchmark.

---

## 2. UTA-RLDD Dataset Composition (Local Fold-1 Subset)

### Dataset Structure
- **Total Processed Videos**: 36 videos across 12 participants (3 videos per participant: Alert `0`, Low Vigilant `5`, Drowsy `10`)
- **Total Generated 16-Frame Sequences**: **20,009 sequences**

### Custom Participant-Level Splits
- **Train Set**: 13,755 sequences across 8 participants (**P01, P02, P03, P06, P09, P10, P11, P12**)
- **Validation Set**: 2,727 sequences across 2 participants (**P05, P08**)
- **Held-Out Test Set**: 3,527 sequences across 2 participants (**P04, P07**)

### Mandatory Benchmark Protocol Disclaimer
> **Methodological Note**: *"The local UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol."*

---

## 3. Split Exclusion & Data Leakage Prevention

- Participants **P04** and **P07** were held out strictly for final test evaluation and diagnostic analysis.
- No sequence from P04 or P07 was exposed to model training or hyperparameter tuning.
