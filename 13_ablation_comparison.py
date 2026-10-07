"""
13_ablation_comparison.py

Runs the ablation the brief asks for:
    BASELINE:  CNN + LSTM + global threshold          (stage 11)
    PROPOSED:  CNN + LSTM + Per-Driver Adaptive Calib. (stage 12)

For every TEST-split driver (never seen in training):
  - use their first CFG.CALIBRATION_SECONDS worth of NORMAL-labeled sequences
    to calibrate a PersonalCalibrator (simulating the webcam calibration step
    offline, using this driver's own held-out data)
  - evaluate both systems on that driver's REMAINING sequences
  - report per-driver and pooled metrics side by side

Also explicitly checks the two failure modes named in the brief:
  - false alarms on drivers whose personal baseline EAR is naturally low
    (narrow eyes) -- would a global threshold over-trigger on them?
  - missed detections on drivers whose personal baseline EAR is naturally
    high (wide eyes) -- would a global threshold under-trigger on them?

Does NOT assume personalization wins -- it just reports both, honestly.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from importlib import import_module

CFG = import_module("00_config").CFG
baseline_lib = import_module("11_baseline_system")
calib_lib = import_module("12_personalized_calibration")


def sequence_mean_ear_mar(npz_cache, npz_path, start, end):
    data = npz_cache.setdefault(npz_path, np.load(npz_path, allow_pickle=True))
    ear = data["ear"][start:end].mean()
    mar = data["mar"][start:end].mean()
    faces = data["faces"][start:end]
    return ear, mar, faces


def build_batch(faces):
    X = np.zeros((1, CFG.SEQ_LEN, CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.float32)
    X[0, :len(faces)] = faces.astype(np.float32)
    return X


def metrics_from(y_true, y_pred):
    if len(set(y_true)) < 2:
        # a driver whose held-out sequences are all one class -- still report what we can
        pass
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "fnr": fnr,
    }


def run_ablation():
    idx_csv = os.path.join(CFG.SEQ_DIR, "sequence_index.csv")
    df = pd.read_csv(idx_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    model = tf.keras.models.load_model(os.path.join(CFG.MODEL_DIR, "cnn_lstm_best.keras"))
    baseline = baseline_lib.BaselineSystem(model)

    npz_cache = {}
    seqs_per_calib = max(CFG.CALIBRATION_MIN_SEQUENCES,
                          int(CFG.CALIBRATION_SECONDS /
                              (CFG.SEQ_LEN * (1 - CFG.SEQ_OVERLAP) / (CFG.NATIVE_FPS / CFG.FRAME_STRIDE))))

    per_driver_rows = []
    baseline_true, baseline_pred = [], []
    personalized_true, personalized_pred = [], []

    for driver_id, group in test_df.groupby("driver_id"):
        group = group.reset_index(drop=True)

        # --- gather raw prob / ear / mar for every sequence of this driver ---
        raw_probs, ears, mars, labels = [], [], [], []
        for _, row in group.iterrows():
            ear, mar, faces = sequence_mean_ear_mar(npz_cache, row["npz_path"],
                                                      row["start_idx"], row["end_idx"])
            prob = baseline.predict_proba(build_batch(faces))[0]
            raw_probs.append(prob); ears.append(ear); mars.append(mar); labels.append(row["label"])

        raw_probs = np.array(raw_probs); ears = np.array(ears); mars = np.array(mars)
        labels = np.array(labels)

        # --- pick calibration sequences: earliest NORMAL (label==0) windows ---
        normal_idx = np.where(labels == 0)[0]
        calib_idx = normal_idx[:seqs_per_calib]
        eval_idx = np.array([i for i in range(len(labels)) if i not in set(calib_idx)])

        if len(calib_idx) < CFG.CALIBRATION_MIN_SEQUENCES or len(eval_idx) == 0:
            print(f"!! skipping driver {driver_id}: not enough normal sequences to calibrate")
            continue

        calibrator = calib_lib.PersonalCalibrator(driver_id=driver_id)
        for i in calib_idx:
            calibrator.add_calibration_sample(raw_probs[i], ears[i], mars[i])
        baseline_info = calibrator.finalize_calibration()

        # --- evaluate both systems on the remaining sequences ---
        b_true, b_pred, p_true, p_pred = [], [], [], []
        for i in eval_idx:
            y = int(labels[i])
            b_dec = baseline.decide(raw_probs[i])
            b_true.append(y); b_pred.append(b_dec)

            p_risk, dev = calibrator.personalized_risk(raw_probs[i], ears[i], mars[i])
            calibrator.maybe_update_baseline(p_risk, ears[i], mars[i])
            p_dec = calibrator.decide(p_risk)
            p_true.append(y); p_pred.append(p_dec)

        baseline_true.extend(b_true); baseline_pred.extend(b_pred)
        personalized_true.extend(p_true); personalized_pred.extend(p_pred)

        row = {
            "driver_id": driver_id,
            "n_eval_sequences": len(eval_idx),
            "baseline_ear_mean": baseline_info["baseline_ear_mean"],
            **{f"baseline_{k}": v for k, v in metrics_from(b_true, b_pred).items() if k != "n"},
            **{f"personalized_{k}": v for k, v in metrics_from(p_true, p_pred).items() if k != "n"},
        }
        per_driver_rows.append(row)

    per_driver_df = pd.DataFrame(per_driver_rows)
    out_csv = os.path.join(CFG.WORK_DIR, "ablation_per_driver.csv")
    per_driver_df.to_csv(out_csv, index=False)

    print("\n=== POOLED (all test drivers) ===")
    print("BASELINE   :", metrics_from(baseline_true, baseline_pred))
    print("PERSONALIZED:", metrics_from(personalized_true, personalized_pred))

    # --- narrow/wide-eye driver check ---
    if len(per_driver_df):
        median_ear = per_driver_df["baseline_ear_mean"].median()
        narrow = per_driver_df[per_driver_df["baseline_ear_mean"] < median_ear * 0.85]
        wide = per_driver_df[per_driver_df["baseline_ear_mean"] > median_ear * 1.15]
        print(f"\nMedian personal baseline EAR across test drivers: {median_ear:.3f}")
        if len(narrow):
            print(f"\nNaturally NARROW-eyed drivers (n={len(narrow)}) -- check for baseline false alarms:")
            print(narrow[["driver_id", "baseline_ear_mean", "baseline_fpr", "personalized_fpr"]])
        if len(wide):
            print(f"\nNaturally WIDE-eyed drivers (n={len(wide)}) -- check for baseline missed detections:")
            print(wide[["driver_id", "baseline_ear_mean", "baseline_fnr", "personalized_fnr"]])

    print(f"\nPer-driver ablation table saved to {out_csv}")
    print("\nReminder: report whichever system actually measures better here -- "
          "do not assume personalization wins without checking these numbers.")
    return per_driver_df


if __name__ == "__main__":
    run_ablation()
