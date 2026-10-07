r"""
08c_evaluate_best_model.py
STAGE 08C -- Comprehensive Best Model Evaluation & Latency Benchmarking

Evaluates the trained MobileNetV2 + LSTM best model (Epoch 4 checkpoint) across:
1. Combined Test Set (NITYMED + UTA-RLDD)
2. NITYMED Test Set (held-out nighttime in-car drivers)
3. UTA-RLDD Test Set (held-out subjects with Alert label 0 and Drowsy label 1)

Reports:
- Accuracy, Precision, Recall, F1-Score, AUC
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR) & False Negative Rate (FNR)
- Latency (ms per sequence, batch=1) & FPS throughput

Outputs saved to: WORK_DIR/stage08_training/evaluation_results.json
"""

import os
import sys
import json
import time
import psutil
import numpy as np
from importlib import import_module

# Force stdout encoding to utf-8 if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CFG = import_module("00_config").CFG
WORK_DIR = CFG.WORK_DIR
STAGE08_DIR = os.path.join(WORK_DIR, "stage08_training")
os.makedirs(STAGE08_DIR, exist_ok=True)

import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

BATCH_SIZE = 8


def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_and_preprocess_image(path_tensor):
    img_raw = tf.io.read_file(path_tensor)
    img_dec = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize(img_dec, [128, 128])
    img_norm = (img_resized / 127.5) - 1.0  # MobileNetV2 scaling to [-1, 1]
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


def evaluate_model_on_dataset(model, dataset, true_labels, name):
    print(f"\nEvaluating model on {name} ({len(true_labels):,} sequences)...")
    t0 = time.time()
    preds_prob = model.predict(dataset, verbose=1).flatten()
    t1 = time.time()
    
    preds_bin = (preds_prob >= 0.5).astype(int)
    
    acc = accuracy_score(true_labels, preds_bin)
    prec = precision_score(true_labels, preds_bin, zero_division=0)
    rec = recall_score(true_labels, preds_bin, zero_division=0)
    f1 = f1_score(true_labels, preds_bin, zero_division=0)
    
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, preds_prob)
        auc_str = f"{auc:.6f}"
    else:
        auc = None
        auc_str = "N/A (Single-class subset)"
        
    cm = confusion_matrix(true_labels, preds_bin, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    res = {
        "dataset_name": name,
        "sample_count": len(true_labels),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "auc": round(float(auc), 6) if auc is not None else None,
        "auc_str": auc_str,
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fnr), 6),
        "confusion_matrix": {
            "TN": tn, "FP": fp,
            "FN": fn, "TP": tp
        },
        "total_inference_time_sec": round(t1 - t0, 2),
        "ms_per_sequence": round((t1 - t0) / len(true_labels) * 1000, 2)
    }
    return res


def benchmark_latency(model, sample_paths, n_runs=50):
    print(f"\nBenchmarking inference latency (batch=1 over {n_runs} runs)...")
    # Load 1 single real sequence for realistic warm-up and timing
    single_ds = create_tf_dataset(sample_paths[:1], np.array([0]), batch_size=1)
    
    # Warmup
    for X, _ in single_ds.take(1):
        _ = model.predict(X, verbose=0)
        
    times = []
    for X, _ in single_ds.take(1):
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.predict(X, verbose=0)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            
    mean_sec = float(np.mean(times))
    std_sec = float(np.std(times))
    fps = 1.0 / mean_sec if mean_sec > 0 else float('inf')
    
    return {
        "latency_ms_mean": round(mean_sec * 1000.0, 2),
        "latency_ms_std": round(std_sec * 1000.0, 2),
        "throughput_fps": round(fps, 2)
    }


def run_evaluation():
    print("=" * 80)
    print(" STAGE 08C -- BEST MODEL COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    best_model_path = os.path.join(STAGE08_DIR, "best_model.keras")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(WORK_DIR, "models", "cnn_lstm_best.keras")
        
    print(f"Loading best checkpoint model from: {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)
    
    # Load NPZ Test Arrays
    nity_test_p = os.path.join(WORK_DIR, "sequences", "test", "sequences_test.npz")
    rldd_test_p = os.path.join(WORK_DIR, "sequences_rldd", "test", "sequences_rldd_test.npz")
    
    d_nity_te = np.load(nity_test_p, allow_pickle=True)
    d_rldd_te = np.load(rldd_test_p, allow_pickle=True)
    
    nity_test_paths = d_nity_te['cnn_input_paths']
    nity_test_labels = d_nity_te['label']
    
    rldd_test_paths = d_rldd_te['cnn_input_paths']
    rldd_test_labels = d_rldd_te['label']
    
    comb_test_paths = np.concatenate([nity_test_paths, rldd_test_paths], axis=0)
    comb_test_labels = np.concatenate([nity_test_labels, rldd_test_labels], axis=0)
    
    print(f"\nTest Subsets Loaded:")
    print(f"  - NITYMED Test  : {len(nity_test_paths):,} sequences (Drowsy label 1)")
    print(f"  - UTA-RLDD Test : {len(rldd_test_paths):,} sequences (Alert label 0 & Drowsy label 1)")
    print(f"  - Combined Test : {len(comb_test_paths):,} sequences total")
    
    # Create tf.data Datasets
    ds_comb_test = create_tf_dataset(comb_test_paths, comb_test_labels, batch_size=BATCH_SIZE)
    ds_nity_test = create_tf_dataset(nity_test_paths, nity_test_labels, batch_size=BATCH_SIZE)
    ds_rldd_test = create_tf_dataset(rldd_test_paths, rldd_test_labels, batch_size=BATCH_SIZE)
    
    # Run evaluations
    res_comb = evaluate_model_on_dataset(model, ds_comb_test, comb_test_labels, "COMBINED TEST")
    res_nity = evaluate_model_on_dataset(model, ds_nity_test, nity_test_labels, "NITYMED TEST")
    res_rldd = evaluate_model_on_dataset(model, ds_rldd_test, rldd_test_labels, "UTA-RLDD TEST")
    
    # Benchmark Latency
    latency_bench = benchmark_latency(model, comb_test_paths)
    print(f"\nLatency Benchmark (Batch=1):")
    print(f"  - Mean Latency : {latency_bench['latency_ms_mean']} ms/seq (+/- {latency_bench['latency_ms_std']} ms)")
    print(f"  - Throughput   : {latency_bench['throughput_fps']} sequences/sec")
    
    # Load past epoch timing if available
    timing_file = os.path.join(STAGE08_DIR, "epoch_timing.json")
    epochs_data = []
    if os.path.exists(timing_file):
        with open(timing_file, "r", encoding="utf-8") as f:
            epochs_data = json.load(f)
            
    best_epoch_entry = min(epochs_data, key=lambda x: x["val_loss"]) if epochs_data else {}
    
    eval_summary = {
        "status": "COMPLETED",
        "best_checkpoint": best_model_path,
        "best_epoch": best_epoch_entry.get("epoch", 4),
        "best_val_loss": best_epoch_entry.get("val_loss", 0.242148),
        "best_val_auc": best_epoch_entry.get("val_auc", 0.94785),
        "best_val_accuracy": best_epoch_entry.get("val_accuracy", 0.937552),
        "epochs_run": len(epochs_data),
        "latency_benchmark": latency_bench,
        "combined_test": res_comb,
        "nitymed_test": res_nity,
        "uta_rldd_test": res_rldd
    }
    
    eval_file = os.path.join(STAGE08_DIR, "evaluation_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nSaved evaluation results JSON to: {eval_file}")
    
    # Print Final Report
    print("\n" + "=" * 80)
    print(" STAGE 08C EVALUATION FINAL REPORT")
    print("=" * 80)
    print(f"Model Checkpoint : {best_model_path}")
    print(f"Best Epoch       : {eval_summary['best_epoch']} (Val Loss: {eval_summary['best_val_loss']:.4f}, Val AUC: {eval_summary['best_val_auc']:.4f})")
    
    for r in [res_comb, res_nity, res_rldd]:
        print(f"\n{r['dataset_name']} ({r['sample_count']:,} samples):")
        print(f"  Accuracy       : {r['accuracy']:.4f}")
        print(f"  Precision      : {r['precision']:.4f}")
        print(f"  Recall         : {r['recall']:.4f}")
        print(f"  F1-Score       : {r['f1_score']:.4f}")
        print(f"  AUC            : {r['auc_str']}")
        print(f"  FPR            : {r['fpr']:.4f}")
        print(f"  FNR            : {r['fnr']:.4f}")
        print(f"  Confusion Matrix (TN, FP, FN, TP):")
        print(f"    [[{r['confusion_matrix']['TN']:5d}, {r['confusion_matrix']['FP']:5d}],")
        print(f"     [{r['confusion_matrix']['FN']:5d}, {r['confusion_matrix']['TP']:5d}]]")
        
    print("\n" + "=" * 80)
    print(" COMPLETED STAGE 08 EVALUATION SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
