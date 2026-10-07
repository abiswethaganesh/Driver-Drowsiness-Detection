# AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)

CNN (MobileNetV2) + LSTM drowsiness classifier on the real NITYMED dataset,
with a Per-Driver Adaptive Calibration module as the novel contribution.

## Confirmed dataset facts used to design this pipeline
Per the dataset's own listing (IEEE DataPort, Petrellis et al., Univ. of
Peloponnese / Aristotle Univ. Thessaloniki): 130 mp4 videos, 25 fps, mute,
11 male + 10 female drivers, captured at night in real cars in Patras, Greece.
- **Yawning** (107 videos): each driver yawns 3 times within a 15-25s clip.
- **Microsleep** (21 videos): ~2 minute clips of the driver talking, looking
  around, and having microsleeps.
- No third "alert-only" category, and no published frame-level timestamps for
  exactly when each yawn/microsleep occurs.

That last point is the whole reason the labeling in stage 06 is a heuristic
EAR/MAR-based pseudo-label rather than a lookup into ground-truth annotation
files — NITYMED doesn't ship those. Treat the labels as a defensible proxy,
not ground truth, and spot-check them.

## Run order (Colab unless noted)

| # | File | What it does |
|---|------|---------------|
| 1 | `01_install_requirements.py` | pip installs |
| 2 | `02_dataset_inspection.py` | scan the real folder/filenames — **run this and check the output before anything else** |
| 3 | `03_video_metadata_generation.py` | build `video_metadata.csv` (category, driver_id, fps, resolution) — adjust `CFG.DRIVER_ID_REGEX` first if stage 2's file tree doesn't match the default assumption |
| 4 | `04_train_val_test_split.py` | driver-level split, asserts zero leakage |
| 5 | `05_frame_extraction.py` | subsampled frames (every 3rd, ~8.3 effective fps) to disk |
| 6 | `06_face_landmark_extraction.py` | face crop + EAR/MAR + pseudo-labels, one `.npz` per video |
| 7 | `07_sequence_generation.py` | builds the 16-frame sliding-window sequence index with sequence-level labels |
| 8 | `08_cnn_lstm_model.py` | model definition (import only, or run for `model.summary()`) |
| 9 | `09_training.py` | trains the CNN-LSTM, saves `cnn_lstm_best.keras` |
| 10 | `10_evaluation.py` | accuracy/precision/recall/F1/CM/FPR/FNR/latency/FPS on the held-out test drivers |
| 11 | `11_baseline_system.py` | wraps the trained model behind a fixed global threshold |
| 12 | `12_personalized_calibration.py` | the novel contribution: `PersonalCalibrator` |
| 13 | `13_ablation_comparison.py` | baseline vs personalized, per-driver, on held-out test drivers |
| 14 | `14_webcam_realtime_inference.py` | **run locally on Windows**, needs a real webcam + display |
| 15 | `15_model_save_load.py` | persistence helpers used by the other stages |

## Key numbers and why

- **Temporal window**: 16 frames sampled every 3rd native frame at 25 fps →
  effective ≈ 8.3 fps → window ≈ **1.92 seconds** of real time. Long enough
  to contain a sustained eye closure or the rise of a yawn; short enough for
  a real-time cadence of roughly one new prediction every second (50% window
  overlap).
- **Labels**: binary "Drowsiness-Indicative Behavior" vs "Normal" per
  ~2-second window, derived from sustained EAR drop (microsleep videos) or
  sustained MAR rise (yawning videos). No 3-class Alert/Partial/Drowsy scheme
  — NITYMED provides no annotation that would justify a "partial" boundary.
- **Splits**: grouped by driver id, never by video or frame, with an
  assertion that fails loudly on any leakage.
- **Personalization**: additive, in logit space, on top of the trained
  model's own probability — never a hand-written threshold replacing the
  network. Baseline updates only from currently low-risk windows, at a slow
  EMA rate, so a driver actually becoming drowsy can't drag their own
  "normal" baseline downward.

## Honesty checks built into the code

- `13_ablation_comparison.py` prints both systems' metrics unconditionally —
  it does not assert personalization is better, and includes an explicit
  breakdown for naturally narrow-eyed vs wide-eyed test drivers so you can
  see whether the false-alarm / missed-detection story the brief asked about
  actually holds up on this data.
- Every stage that makes an assumption about file naming, thresholds, or
  dataset structure prints a warning if that assumption looks violated
  (e.g. unmatched driver ids, "unknown" category, too few calibration
  sequences for a driver).

## Known limitations to disclose in your report

1. Pseudo-labels are EAR/MAR-threshold heuristics, not dataset-provided
   ground truth — NITYMED does not ship per-frame annotations.
2. 21 microsleep videos is a small pool of drivers for that behavior; expect
   higher variance in microsleep-specific metrics than yawning ones.
3. `DRIVER_ID_REGEX` is a guess at NITYMED's naming convention — verify it
   against the real filenames from stage 2 before trusting the driver-level
   split.
