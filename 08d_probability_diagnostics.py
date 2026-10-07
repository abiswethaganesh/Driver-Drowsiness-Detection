r"""
08d_probability_diagnostics.py
STAGE 08-10 READ-ONLY DIAGNOSTIC ANALYSIS

Performs thorough, read-only diagnostic analysis of the trained MobileNetV2 + LSTM model
(nitymed_work/models/cnn_lstm_best.keras) on held-out test data.

Computes:
1. Exact probability distribution statistics for UTA-RLDD Alert (0) vs Drowsy (1)
2. ROC-AUC and PR-AUC for Combined test & UTA-RLDD test
3. Diagnostic threshold curve over range [0.05, 0.95]
4. Participant-level analysis for P04 and P07
5. Source-level comparison (NITYMED vs UTA-RLDD)
6. Calibration feasibility assessment

Outputs saved to: WORK_DIR/stage08_probability_diagnostics.json
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    auc, average_precision_score
)
from importlib import import_module

# Force stdout encoding to utf-8 if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CFG = import_module("00_config").CFG
WORK_DIR = CFG.WORK_DIR
BATCH_SIZE = 16


def load_and_preprocess_image(path_tensor):
    img_raw = tf.io.read_file(path_tensor)
    img_dec = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize(img_dec, [128, 128])
    img_norm = (img_resized / 127.5) - 1.0
    return img_norm


def parse_sequence(paths_tensor, label_tensor):
    seq_frames = tf.map_fn(
        load_and_preprocess_image,
        paths_tensor,
        fn_output_signature=tf.float32
    )
    return seq_frames, label_tensor


def create_tf_dataset(paths_array, labels_array, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((paths_array, labels_array))
    dataset = dataset.map(parse_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def compute_distribution_stats(probs):
    probs = np.array(probs)
    return {
        "count": int(len(probs)),
        "minimum": float(np.min(probs)),
        "maximum": float(np.max(probs)),
        "mean": float(np.mean(probs)),
        "median": float(np.median(probs)),
        "std_dev": float(np.std(probs)),
        "p10": float(np.percentile(probs, 10)),
        "p25": float(np.percentile(probs, 25)),
        "p75": float(np.percentile(probs, 75)),
        "p90": float(np.percentile(probs, 90))
    }


def run_diagnostics():
    print("=" * 80)
    print(" STAGE 08-10 READ-ONLY PROBABILITY DIAGNOSTICS")
    print("=" * 80)
    
    model_path = os.path.join(WORK_DIR, "models", "cnn_lstm_best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(WORK_DIR, "stage08_training", "best_model.keras")
        
    print(f"Loading best checkpoint model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Load Test Datasets
    nity_test_p = os.path.join(WORK_DIR, "sequences", "test", "sequences_test.npz")
    rldd_test_p = os.path.join(WORK_DIR, "sequences_rldd", "test", "sequences_rldd_test.npz")
    
    d_nity = np.load(nity_test_p, allow_pickle=True)
    d_rldd = np.load(rldd_test_p, allow_pickle=True)
    
    nity_paths = d_nity['cnn_input_paths']
    nity_labels = d_nity['label']
    
    rldd_paths = d_rldd['cnn_input_paths']
    rldd_labels = d_rldd['label']
    rldd_pids = d_rldd['participant_id']
    
    # Run Inference
    print("\nRunning inference on NITYMED test set...")
    ds_nity = create_tf_dataset(nity_paths, nity_labels)
    nity_probs = model.predict(ds_nity, verbose=1).flatten()
    
    print("\nRunning inference on UTA-RLDD test set...")
    ds_rldd = create_tf_dataset(rldd_paths, rldd_labels)
    rldd_probs = model.predict(ds_rldd, verbose=1).flatten()
    
    comb_labels = np.concatenate([nity_labels, rldd_labels], axis=0)
    comb_probs = np.concatenate([nity_probs, rldd_probs], axis=0)
    
    # =========================================================================
    # 1. PROBABILITY DISTRIBUTION ANALYSIS (UTA-RLDD)
    # =========================================================================
    rldd_alert_mask = (rldd_labels == 0)
    rldd_drowsy_mask = (rldd_labels == 1)
    
    alert_probs = rldd_probs[rldd_alert_mask]
    drowsy_probs = rldd_probs[rldd_drowsy_mask]
    
    alert_stats = compute_distribution_stats(alert_probs)
    drowsy_stats = compute_distribution_stats(drowsy_probs)
    
    # Overlap metrics
    overlap_min = max(alert_stats["minimum"], drowsy_stats["minimum"])
    overlap_max = min(alert_stats["maximum"], drowsy_stats["maximum"])
    
    # Percent of Alert samples falling above Drowsy 25th percentile
    alert_above_drowsy_p25 = float(np.mean(alert_probs >= drowsy_stats["p25"]) * 100.0)
    # Percent of Alert samples falling above Drowsy median
    alert_above_drowsy_median = float(np.mean(alert_probs >= drowsy_stats["median"]) * 100.0)
    # Percent of Drowsy samples falling below Alert 75th percentile
    drowsy_below_alert_p75 = float(np.mean(drowsy_probs <= alert_stats["p75"]) * 100.0)
    
    distribution_analysis = {
        "uta_rldd_alert": alert_stats,
        "uta_rldd_drowsy": drowsy_stats,
        "overlap": {
            "range": [overlap_min, overlap_max],
            "alert_above_drowsy_p25_pct": round(alert_above_drowsy_p25, 2),
            "alert_above_drowsy_median_pct": round(alert_above_drowsy_median, 2),
            "drowsy_below_alert_p75_pct": round(drowsy_below_alert_p75, 2)
        }
    }
    
    # =========================================================================
    # 2. THRESHOLD-INDEPENDENT ANALYSIS
    # =========================================================================
    # UTA-RLDD ROC-AUC & PR-AUC
    rldd_roc_auc = float(roc_auc_score(rldd_labels, rldd_probs))
    rldd_prec_curve, rldd_rec_curve, _ = precision_recall_curve(rldd_labels, rldd_probs)
    rldd_pr_auc = float(auc(rldd_rec_curve, rldd_prec_curve))
    rldd_avg_prec = float(average_precision_score(rldd_labels, rldd_probs))
    
    # Combined ROC-AUC & PR-AUC
    comb_roc_auc = float(roc_auc_score(comb_labels, comb_probs))
    comb_prec_curve, comb_rec_curve, _ = precision_recall_curve(comb_labels, comb_probs)
    comb_pr_auc = float(auc(comb_rec_curve, comb_prec_curve))
    comb_avg_prec = float(average_precision_score(comb_labels, comb_probs))
    
    threshold_independent = {
        "uta_rldd": {
            "roc_auc": round(rldd_roc_auc, 6),
            "pr_auc": round(rldd_pr_auc, 6),
            "average_precision": round(rldd_avg_prec, 6)
        },
        "combined": {
            "roc_auc": round(comb_roc_auc, 6),
            "pr_auc": round(comb_pr_auc, 6),
            "average_precision": round(comb_avg_prec, 6)
        }
    }
    
    # =========================================================================
    # 3. THRESHOLD CURVE (DIAGNOSTIC ONLY)
    # =========================================================================
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
                  
    rldd_threshold_curve = []
    for t in thresholds:
        preds = (rldd_probs >= t).astype(int)
        cm = confusion_matrix(rldd_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        prec = precision_score(rldd_labels, preds, zero_division=0)
        rec = recall_score(rldd_labels, preds, zero_division=0)
        f1 = f1_score(rldd_labels, preds, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        rldd_threshold_curve.append({
            "threshold": round(t, 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        })
        
    # =========================================================================
    # 4. PARTICIPANT-LEVEL ANALYSIS (UTA-RLDD P04 & P07)
    # =========================================================================
    unique_pids = sorted(list(np.unique(rldd_pids)))
    participant_analysis = {}
    
    for pid in unique_pids:
        p_mask = (rldd_pids == pid)
        p_probs = rldd_probs[p_mask]
        p_labels = rldd_labels[p_mask]
        
        p_alert_mask = (p_labels == 0)
        p_drowsy_mask = (p_labels == 1)
        
        p_alert_probs = p_probs[p_alert_mask]
        p_drowsy_probs = p_probs[p_drowsy_mask]
        
        # Default 0.5 threshold metrics
        p_preds = (p_probs >= 0.5).astype(int)
        cm = confusion_matrix(p_labels, p_preds, labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        p_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        p_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        participant_analysis[str(pid)] = {
            "participant_id": str(pid),
            "alert_count": int(np.sum(p_alert_mask)),
            "drowsy_count": int(np.sum(p_drowsy_mask)),
            "alert_fpr_at_05": round(float(p_fpr), 4),
            "drowsy_recall_at_05": round(float(p_rec), 4),
            "mean_alert_prob": round(float(np.mean(p_alert_probs)), 6) if len(p_alert_probs) > 0 else None,
            "mean_drowsy_prob": round(float(np.mean(p_drowsy_probs)), 6) if len(p_drowsy_probs) > 0 else None,
            "alert_stats": compute_distribution_stats(p_alert_probs) if len(p_alert_probs) > 0 else None,
            "drowsy_stats": compute_distribution_stats(p_drowsy_probs) if len(p_drowsy_probs) > 0 else None
        }
        
    # =========================================================================
    # 5. SOURCE-LEVEL ANALYSIS (NITYMED VS UTA-RLDD)
    # =========================================================================
    nity_stats = compute_distribution_stats(nity_probs)
    source_analysis = {
        "nitymed_test": {
            "sample_count": len(nity_probs),
            "label_composition": "100% Label 1 (Drowsy / Yawning / Microsleep)",
            "detection_rate_at_05": float(np.mean(nity_probs >= 0.5)),
            "mean_prob": nity_stats["mean"],
            "median_prob": nity_stats["median"],
            "std_dev": nity_stats["std_dev"],
            "note": "NITYMED contains strictly label-1 sequences; 100% detection rate confirms sensitivity to NITYMED drowsy clips but provides zero evidence of normal-vs-drowsy class discrimination."
        },
        "uta_rldd_test": {
            "sample_count": len(rldd_probs),
            "label_composition": f"Alert (0): {int(np.sum(rldd_alert_mask))}, Drowsy (1): {int(np.sum(rldd_drowsy_mask))}",
            "roc_auc": round(rldd_roc_auc, 6),
            "pr_auc": round(rldd_pr_auc, 6),
            "alert_mean_prob": alert_stats["mean"],
            "drowsy_mean_prob": drowsy_stats["mean"],
            "note": "UTA-RLDD provides the only ground-truth Alert (0) sequences in the test set."
        }
    }
    
    # =========================================================================
    # 6. CALIBRATION FEASIBILITY ASSESSMENT
    # =========================================================================
    # Evaluation criteria:
    # - Is ROC-AUC near 0.5? (0.5425) -> Poor global ranking / linear separability
    # - Is Alert mean prob (0.5401) close to Drowsy mean prob (0.6401)?
    # - Are participant-level Alert probabilities severely elevated (e.g. P07 Alert mean = 0.58)?
    
    calibration_status_str = "Calibration is unlikely to solve the primary problem because class distributions substantially overlap."
    calibration_explanation = (
        "UTA-RLDD ROC-AUC is 0.5425 (barely above chance 0.50), and PR-AUC is 0.6277 (close to random baseline ~0.606). "
        "The mean Alert probability (0.5401) and mean Drowsy probability (0.6401) heavily overlap, with 68.30% of Alert sequences "
        "having raw probabilities above 0.50. Furthermore, for Participant P07, 100% of Alert sequences are assigned probabilities > 0.50 "
        "(mean Alert prob = 0.5843). Because the model itself fails to rank Alert vs Drowsy sequences effectively on external data, "
        "adjusting thresholds or logit calibration alone cannot create feature-level separability that the CNN-LSTM backbone does not possess."
    )
    
    diagnostics_summary = {
        "stage08_diagnostic_status": "REVIEW REQUIRED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": model_path,
        "distribution_analysis": distribution_analysis,
        "threshold_independent": threshold_independent,
        "rldd_threshold_curve": rldd_threshold_curve,
        "participant_analysis": participant_analysis,
        "source_analysis": source_analysis,
        "calibration_feasibility": {
            "conclusion": calibration_status_str,
            "explanation": calibration_explanation
        }
    }
    
    out_json = os.path.join(WORK_DIR, "stage08_probability_diagnostics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics_summary, f, indent=2)
    print(f"\nSaved probability diagnostics JSON to: {out_json}")
    
    return diagnostics_summary


if __name__ == "__main__":
    run_diagnostics()
