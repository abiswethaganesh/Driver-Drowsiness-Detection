# STAGE 08B — UTA-RLDD External Dataset Integration Plan

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)  
**Document**: Technical Integration Plan (`08b_uta_rldd_integration_plan.md`)  
**Status**: Pre-Download Architecture & Ingestion Design  

---

## 1. Executive Summary & Objectives

This document establishes the detailed implementation specification for integrating the **University of Texas at Arlington Real-Life Drowsiness Dataset (UTA-RLDD)** as an external dataset to supply genuine ground-truth **Normal / Alert (label 0)** and **Drowsy (label 1)** sequences.

### Primary Objectives
1. **Supply Ground-Truth Normal Class (`label 0`)**: Obtain explicit alert driving sequences to complement NITYMED's drowsiness-indicative clips.
2. **Prevent Dataset-Source Shortcut Learning**: Incorporate both **Alert (`label 0`)** and **Drowsy (`label 1`)** samples from UTA-RLDD alongside NITYMED's Drowsy samples, preventing the MobileNetV2 CNN from exploiting background or camera sensor artifacts.
3. **Preserve NITYMED Split Integrity**: Keep NITYMED's existing 126 local videos and 5,822 sequences across Stage 04 video-level splits (4,118 train, 860 val, 844 test) completely untouched.
4. **Enforce Strict Subject-Level Splitting for UTA-RLDD**: Partition UTA-RLDD participants exclusively by subject ID to prevent data leakage.

---

## 2. Official UTA-RLDD Source & Manual Download Procedure

### Official Source Verification
- **Official Webpage**: `https://sites.google.com/view/utarldd/home`
- **Primary Publication**: *Real-Life Drowsiness Dataset (RLDD) for Multi-Stage Drowsiness Detection* (F. Silva, V. Metsis et al., 2019, arXiv:1904.07312).
- **License**: Publicly available for academic research (CC0 Public Domain / CC BY-NC-SA 4.0 for derived mirrors).

### Manual Download Procedure (Step-by-Step)
1. **Target Local Ingestion Directory**:  
   `D:\Sem 7\NNDL\Project Demo\UTA_RLDD_raw\` (or configured `CFG.UTA_RLDD_ROOT`).
2. **Download Protocol**:
   - Access the official project Google Drive link or primary academic archive repository.
   - Download the raw video archives containing the 60 participant folders.
   - Extract raw video zip archives directly into `UTA_RLDD_raw/`.
3. **Post-Download Inspection**:
   - Verify folder names, participant directory formats, video container extensions (`.mp4`, `.avi`, `.mov`), and file counts prior to executing any script.

---

## 3. Expected Directory & File Structure

Based on published author specifications (Silva et al., 2019), UTA-RLDD contains 180 video clips recorded from 60 participants (3 videos per participant corresponding to 3 target states).

```text
UTA_RLDD_raw/
├── 01/
│   ├── 0.mp4    (Alert session)
│   ├── 5.mp4    (Low Vigilant session)
│   └── 10.mp4   (Drowsy session)
├── 02/
│   ├── 0.mp4
│   ├── 5.mp4
│   └── 10.mp4
...
└── 60/
    ├── 0.mp4
    ├── 5.mp4
    └── 10.mp4
```

*Note: The ingestion script will inspect the directory dynamically upon execution to account for naming variations (e.g. `Participant_01`, `01_0.mp4`, etc.) without hardcoding assumptions.*

---

## 4. Participant ID & Label Mapping Protocol

### Participant ID Detection
- Participant IDs will be extracted from folder names or filenames strictly after raw directory inspection (e.g. `P01` to `P60` or integer folder names `1` to `60`).
- No participant identity will be inferred from arbitrary file attributes.

### Label & Behavior Mapping

| UTA-RLDD Class / Filename | Documented State | Assigned Binary Label | Assigned Behavior Name | Included in Ingestion? |
|---|---|---|---|---|
| **Class 0** (`0.mp4` / Alert) | Completely conscious & alert | **`label = 0`** | `Alert` | **YES (Primary Normal Source)** |
| **Class 5** (`5.mp4` / Low Vigilant) | Subtle sleepiness signs | **`label = 1`** | `LowVigilant` | **YES (External Drowsy Source)** |
| **Class 10** (`10.mp4` / Drowsy) | Actively struggling with sleep | **`label = 1`** | `Drowsy` | **YES (External Drowsy Source)** |

---

## 5. Subject-Level Train / Validation / Test Splitting Strategy

### Split Rules & Ratios
To guarantee zero data leakage:
- **Participant-Level Partitioning**: All videos and frames belonging to a single participant MUST reside in exactly one split.
- **Target Ratios**:
  - **Train**: ~70% of subjects (~42 subjects)
  - **Validation**: ~15% of subjects (~9 subjects)
  - **Test**: ~15% of subjects (~9 subjects)
- **Random Seed**: Fixed random seed (`RANDOM_SEED = 42`) for reproducibility.

### Participant Split Mapping

```text
60 Total Participants (Random Seed 42)
├── Train Split (42 subjects)      --> All videos/sequences to sequences_rldd_train.npz
├── Validation Split (9 subjects)  --> All videos/sequences to sequences_rldd_val.npz
└── Test Split (9 subjects)        --> All videos/sequences to sequences_rldd_test.npz
```

---

## 6. Frame Sampling & Subsampling Strategy

To ensure temporal alignment with NITYMED:
- **Native FPS**: UTA-RLDD videos are recorded at varying rates ($<30$ FPS).
- **Effective Stride**: Subsample frames using `FRAME_STRIDE = 3` (matching NITYMED's effective ~8.33 FPS sampling rate).
- **Resolution**: Extract full-frame RGB images to `WORK_DIR/rldd_frames/{split}/{behavior}/{video_stem}/`.

---

## 7. Face Detection & Landmark Extraction Strategy (Stage 06 Alignment)

- **Landmark Engine**: Google MediaPipe `FaceLandmarker` (`face_landmarker.task`).
- **Cropping Protocol**: Extract $128 \times 128$ tight face crops with bounding box expansion factor $0.25$ (identical to Stage 06).
- **Storage**: Save cropped JPEGs to `WORK_DIR/rldd_faces/{split}/{behavior}/{video_stem}/`.
- **Quality Check**: Store landmark availability (`landmark_available == True/False`) per frame in `WORK_DIR/rldd_face_landmark_metadata.csv`.

---

## 8. 16-Frame Temporal Sequence Generation Strategy (Stage 07 Alignment)

- **Sequence Window**: `SEQ_LEN = 16` consecutive frames per sliding window.
- **Overlap**: `SEQ_OVERLAP = 0.5` (stride of 8 frames).
- **Valid Sequence Threshold**: Sequence retained only if $\ge 80\%$ ( $\ge 13 / 16$ ) of frames have valid face landmarks.
- **Missing Frame Handling**: For 1–3 missing face crops in a valid sequence, assign nearest valid face crop path to `cnn_input_path` while keeping original `landmark_available` booleans intact and logging `reused_frame_count`.

---

## 9. Metadata Schema & Dataset Source Tracking

### Unified Sequence Metadata CSV Schema (`WORK_DIR/rldd_sequence_metadata.csv`)

| Field Name | Type | Description / Value |
|---|---|---|
| `sequence_id` | String | e.g. `seq_rldd_train_000001` |
| `dataset_source` | String | **`"UTA-RLDD"`** (Mandatory for tracking) |
| `participant_id` | String | e.g. `P01` (Extracted from official structure) |
| `video_filename` | String | e.g. `P01_0.mp4` |
| `split` | String | `train` / `val` / `test` |
| `behavior` | String | `Alert` / `LowVigilant` / `Drowsy` |
| `label` | Int | `0` (Alert) or `1` (LowVigilant / Drowsy) |
| `start_frame_index` | Int | Starting frame index |
| `end_frame_index` | Int | Ending frame index |
| `seq_len` | Int | `16` |
| `valid_frame_count` | Int | Count of valid landmark frames (13–16) |
| `reused_frame_count` | Int | Count of substituted face crops |
| `duration_seconds` | Float | `(end_frame_index - start_frame_index) / source_fps` |

---

## 10. Combined Dataset Architecture & Multi-Source Evaluation Protocol

### Isolated Sequence Storage Strategy
To prevent overwriting existing files:
- NITYMED sequences remain in `WORK_DIR/sequences/{split}/sequences_{split}.npz`.
- UTA-RLDD sequences will be saved to `WORK_DIR/sequences_rldd/{split}/sequences_rldd_{split}.npz`.

### Combined Logical Dataset Setup for Stage 08 / Stage 09

```text
Combined Training Set:
├── NITYMED Train Sequences    (dataset_source = "NITYMED", label = 1)
└── UTA-RLDD Train Sequences   (dataset_source = "UTA-RLDD", label = 0 & 1)

Combined Validation Set:
├── NITYMED Val Sequences      (dataset_source = "NITYMED", label = 1)
└── UTA-RLDD Val Sequences     (dataset_source = "UTA-RLDD", label = 0 & 1)

Combined Test Set (Isolated Reporting):
├── NITYMED Test Sequences    (dataset_source = "NITYMED", label = 1)
└── UTA-RLDD Test Sequences   (dataset_source = "UTA-RLDD", label = 0 & 1)
```

### Evaluation Reporting Protocol (Stage 10)
Stage 10 (`10_evaluation.py`) will load combined test data and report performance breakdowns across three distinct views:
1. **Overall Combined Test Performance** (Accuracy, Precision, Recall, F1, FPR, FNR)
2. **NITYMED Test Performance** (Measuring drowsiness recall on nighttime in-car NIR data)
3. **UTA-RLDD Test Performance** (Measuring alert specificity and drowsiness recall on external data)

---

## 11. Data Leakage Safeguards & Validation Matrix

The ingestion script (`07b_external_sequence_generation.py`) will execute an automated 10-point validation suite:

1. **Subject Isolation Check**: Assert zero overlap of `participant_id` across `train`, `val`, and `test` splits.
2. **Video Isolation Check**: Assert zero video filename overlap across splits.
3. **Sequence Isolation Check**: Assert zero sequence ID duplicates.
4. **Frame Continuity Check**: Assert strictly increasing frame indices per sequence.
5. **Exact Sequence Length Check**: Assert `seq_len == 16` for all sequences.
6. **Boundary Check**: Assert no sequence crosses participant or video boundaries.
7. **NITYMED Preservation Check**: Assert NITYMED Stage 04 CSV files and NPZ archives are completely unmodified.
8. **File Existence Check**: Assert all frame crop paths in NPZ matrices exist on disk.
9. **Label Consistency Check**: Assert `Alert` maps to `0` and `LowVigilant/Drowsy` map to `1`.
10. **Source Metadata Check**: Assert `dataset_source == "UTA-RLDD"` across all metadata rows.

---

## 12. Class Balance & Weighting Strategy

- **No Artificial Duplication**: Sequences will NOT be duplicated to force an arbitrary 50/50 ratio.
- **Empirical Ratio Calculation**: Following dataset extraction, exact sequence counts per split and label will be computed.
- **Training Loss Weighting**: Class imbalance during Stage 09 training will be handled dynamically using PyTorch/TensorFlow class weights:
  $$W_{c} = \frac{N_{\text{total}}}{2 \cdot N_{c}}$$
  where $N_{c}$ is the sequence count for class $c \in \{0, 1\}$.

---

## 13. Real-Time Webcam Demonstration Compatibility

The integration of UTA-RLDD is strictly for dataset-level training and validation split integrity.

### Real-Time Webcam Pipeline (Stage 14)
The live webcam inference pipeline remains entirely agnostic to dataset sources or participant IDs:
$$\text{Webcam Stream} \rightarrow \text{MediaPipe FaceLandmarker Crop} \rightarrow \text{MobileNetV2} \rightarrow \text{LSTM} \rightarrow P(\text{Drowsy}) \rightarrow \text{Severity} \rightarrow \text{Intervention} \rightarrow \text{Recovery Verification}$$

---

## 14. Summary & Next Steps

### Action Checklist
- [x] Complete technical audit and integration plan (`08b_uta_rldd_integration_plan.md`).
- [ ] User approval of integration plan.
- [ ] Manual download of UTA-RLDD raw zip archive into `UTA_RLDD_raw/`.
- [ ] Execution of dynamic dataset inspection script (`02b_uta_rldd_inspection.py`).
- [ ] Execution of external sequence pipeline script (`07b_external_sequence_generation.py`).
- [ ] Proceed to Stage 08 (CNN-LSTM Model Definition).

---
*End of Integration Plan (`08b_uta_rldd_integration_plan.md`)*
