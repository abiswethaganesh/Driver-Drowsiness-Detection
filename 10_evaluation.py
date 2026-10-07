"""
10_evaluation.py
Evaluates the trained CNN-LSTM model on the held-out TEST split (entirely
unseen drivers, per stage 04). Reports the metrics requested in the brief,
plus inference latency and achievable FPS for the real-time pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from importlib import import_module

CFG = import_module("00_config").CFG
train_lib = import_module("09_training")


def load_model(path=None):
    path = path or os.path.join(CFG.MODEL_DIR, "cnn_lstm_best.keras")
    return tf.keras.models.load_model(path)


def get_test_arrays(model, test_df):
    gen = train_lib.SequenceGenerator(test_df, batch_size=CFG.BATCH_SIZE, shuffle=False)
    y_true, y_prob = [], []
    for i in range(len(gen)):
        X, y = gen[i]
        p = model.predict(X, verbose=0).ravel()
        y_true.extend(y.tolist())
        y_prob.extend(p.tolist())
    return np.array(y_true), np.array(y_prob)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "fpr": fpr,
        "fnr": fnr,
    }


def benchmark_latency(model, n_runs=50):
    dummy = np.random.rand(1, CFG.SEQ_LEN, CFG.IMG_SIZE, CFG.IMG_SIZE, 3).astype(np.float32) * 255
    model.predict(dummy, verbose=0)  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append(time.perf_counter() - t0)
    mean_latency = float(np.mean(times))
    fps = 1.0 / mean_latency if mean_latency > 0 else float("inf")
    return mean_latency, fps


def print_metrics(name, m):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1-score : {m['f1']:.4f}")
    print(f"FPR      : {m['fpr']:.4f}")
    print(f"FNR      : {m['fnr']:.4f}")
    print(f"Confusion matrix [[TN FP][FN TP]]:\n{m['confusion_matrix']}")


def main():
    idx_csv = os.path.join(CFG.SEQ_DIR, "sequence_index.csv")
    df = pd.read_csv(idx_csv)
    test_df = df[df["split"] == "test"]

    model = load_model()
    y_true, y_prob = get_test_arrays(model, test_df)

    metrics = compute_metrics(y_true, y_prob, CFG.GLOBAL_THRESHOLD)
    print_metrics(f"Test set (global threshold={CFG.GLOBAL_THRESHOLD})", metrics)

    latency, fps = benchmark_latency(model)
    print(f"\nInference latency (single sequence, batch=1): {latency*1000:.1f} ms")
    print(f"Achievable throughput: {fps:.1f} sequences/sec")
    print("Note: at CFG.SEQ_LEN frames per sequence with 50% overlap, a new "
          f"prediction is needed roughly every {CFG.SEQ_LEN * (1-CFG.SEQ_OVERLAP) / (CFG.NATIVE_FPS/CFG.FRAME_STRIDE):.2f}s "
          "of video time for the real-time demo -- compare against that, not raw camera FPS.")

    # Save raw predictions for later reuse (e.g. by the ablation script)
    out = test_df.copy()
    out["y_true"] = y_true
    out["y_prob"] = y_prob
    out.to_csv(os.path.join(CFG.SEQ_DIR, "test_predictions.csv"), index=False)
    print(f"\nSaved per-sequence test predictions to "
          f"{os.path.join(CFG.SEQ_DIR, 'test_predictions.csv')}")


if __name__ == "__main__":
    main()
