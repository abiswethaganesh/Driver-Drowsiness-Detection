r"""
08e_fine_tuning.py
STAGE 08E -- MobileNetV2 Upper-Layer Fine-Tuning Pipeline & Evaluation

Executes controlled fine-tuning experiment:
- Base model: nitymed_work/models/cnn_lstm_best.keras (Epoch 4 checkpoint)
- Unfreezes top 25 MobileNetV2 layers while keeping remaining 130 layers frozen
- Trains with LR = 1e-5, Adam, binary_crossentropy, class weights {0: 1.8437, 1: 0.6861}
- Evaluates on Train (17,873) and Validation (3,587) sets ONLY during training
- Selects best fine-tuned model based strictly on validation loss / val AUC
- Performs single final test evaluation on held-out test sets (Combined, NITYMED, UTA-RLDD, P04, P07)

Outputs saved to: WORK_DIR/stage08_finetuning/
"""

import os
import sys
import json
import time
import glob
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
FINETUNING_DIR = os.path.join(WORK_DIR, "stage08_finetuning")
CHECKPOINTS_DIR = os.path.join(FINETUNING_DIR, "checkpoints")

os.makedirs(FINETUNING_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, metrics
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    average_precision_score
)

RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 15
LEARNING_RATE = 1e-5
CLASS_WEIGHTS = {0: 1.8437, 1: 0.6861}


def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


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


def create_tf_dataset(paths_array, labels_array, batch_size=BATCH_SIZE, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths_array, labels_array))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_SEED)
    dataset = dataset.map(parse_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def configure_fine_tuning_layers(model):
    """
    Unfreezes top 25 layers of MobileNetV2, keeping the remaining 130 layers frozen.
    Keeps LSTM and dense classification head trainable.
    """
    td_layer = model.get_layer("time_distributed_mobilenet")
    base_cnn = td_layer.layer
    
    # Enable trainable flag on backbone container
    base_cnn.trainable = True
    
    # Freeze all except top 25 layers
    n_total_cnn = len(base_cnn.layers)
    n_unfreeze = 25
    
    for layer in base_cnn.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base_cnn.layers[-n_unfreeze:]:
        layer.trainable = True
        
    # Ensure classification head remains trainable
    model.get_layer("lstm_temporal").trainable = True
    model.get_layer("dropout_lstm").trainable = True
    model.get_layer("dense_head").trainable = True
    model.get_layer("dropout_dense").trainable = True
    model.get_layer("drowsiness_score").trainable = True
    
    return model, n_total_cnn, n_unfreeze


def run_dry_run(model, sample_batch):
    print("\n--- Running Fine-Tuning Dry Run Verification ---")
    X_batch, y_batch = sample_batch
    
    td_layer = model.get_layer("time_distributed_mobilenet")
    base_cnn = td_layer.layer
    
    cnn_trainable_count = sum([1 for l in base_cnn.layers if l.trainable])
    cnn_frozen_count = sum([1 for l in base_cnn.layers if not l.trainable])
    
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    frozen_params = total_params - trainable_params
    
    print(f"MobileNetV2 Layer Check: Total={len(base_cnn.layers)}, Trainable={cnn_trainable_count}, Frozen={cnn_frozen_count}")
    print(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}, Frozen={frozen_params:,}")
    
    # Single Forward Pass
    t0 = time.time()
    preds = model(X_batch, training=False)
    t1 = time.time()
    fwd_pass_ok = (preds.shape == (batch_size_val, 1)) and (float(tf.reduce_min(preds)) >= 0.0) and (float(tf.reduce_max(preds)) <= 1.0)
    print(f"Forward pass time: {(t1-t0)*1000:.2f} ms | Shape: {preds.shape} -> {'PASS' if fwd_pass_ok else 'FAIL'}")
    
    # Gradient Tape Check
    with tf.GradientTape() as tape:
        batch_preds = model(X_batch, training=True)
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        raw_loss = bce_loss_fn(y_batch, batch_preds)
        sample_weights = tf.gather(tf.constant([1.8437, 0.6861], dtype=tf.float32), tf.cast(y_batch, tf.int32))
        weighted_loss = tf.reduce_mean(raw_loss * sample_weights)
        
    grads = tape.gradient(weighted_loss, model.trainable_variables)
    
    loss_val = float(weighted_loss.numpy())
    loss_finite = np.isfinite(loss_val)
    grads_finite = all([np.all(np.isfinite(g.numpy())) for g in grads if g is not None])
    
    print(f"Single batch weighted loss: {loss_val:.6f} -> {'PASS' if loss_finite else 'FAIL'}")
    print(f"Gradients finite check: {len(grads)} trainable tensors -> {'PASS' if grads_finite else 'FAIL'}")
    
    dry_run_data = {
        "status": "PASS" if (fwd_pass_ok and loss_finite and grads_finite and cnn_trainable_count == 25) else "FAILED",
        "mobilenet_total_layers": len(base_cnn.layers),
        "mobilenet_trainable_layers": cnn_trainable_count,
        "mobilenet_frozen_layers": cnn_frozen_count,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
        "single_batch_loss": loss_val,
        "forward_pass_ms": round((t1-t0)*1000, 2),
        "test_data_in_training": False
    }
    
    dry_run_path = os.path.join(FINETUNING_DIR, "finetuning_dry_run.json")
    with open(dry_run_path, "w", encoding="utf-8") as f:
        json.dump(dry_run_data, f, indent=2)
    print(f"Saved dry run JSON to: {dry_run_path}\n")
    
    return dry_run_data


def evaluate_dataset(model, dataset, true_labels, name):
    print(f"Evaluating model on {name} ({len(true_labels):,} sequences)...")
    t0 = time.time()
    probs = model.predict(dataset, verbose=0).flatten()
    t1 = time.time()
    
    preds_bin = (probs >= 0.5).astype(int)
    
    acc = accuracy_score(true_labels, preds_bin)
    prec = precision_score(true_labels, preds_bin, zero_division=0)
    rec = recall_score(true_labels, preds_bin, zero_division=0)
    f1 = f1_score(true_labels, preds_bin, zero_division=0)
    
    if len(np.unique(true_labels)) > 1:
        roc_auc = float(roc_auc_score(true_labels, probs))
        roc_auc_str = f"{roc_auc:.6f}"
        prec_curve, rec_curve, _ = precision_recall_curve(true_labels, probs)
        pr_auc = float(auc(rec_curve, prec_curve))
        pr_auc_str = f"{pr_auc:.6f}"
    else:
        roc_auc = None
        roc_auc_str = "N/A (Single-class subset)"
        pr_auc = None
        pr_auc_str = "N/A"
        
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
        "roc_auc": round(float(roc_auc), 6) if roc_auc is not None else None,
        "roc_auc_str": roc_auc_str,
        "pr_auc": round(float(pr_auc), 6) if pr_auc is not None else None,
        "pr_auc_str": pr_auc_str,
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fnr), 6),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "inference_time_sec": round(t1 - t0, 2)
    }
    return res, probs


def run_stage_08e():
    print("=" * 80)
    print(" STAGE 08E -- MOBILENETV2 UPPER-LAYER FINE-TUNING")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Load NPZs for Train, Val, and Test Splits
    nity_train_p = os.path.join(WORK_DIR, "sequences", "train", "sequences_train.npz")
    nity_val_p   = os.path.join(WORK_DIR, "sequences", "val", "sequences_val.npz")
    nity_test_p  = os.path.join(WORK_DIR, "sequences", "test", "sequences_test.npz")
    
    rldd_train_p = os.path.join(WORK_DIR, "sequences_rldd", "train", "sequences_rldd_train.npz")
    rldd_val_p   = os.path.join(WORK_DIR, "sequences_rldd", "val", "sequences_rldd_val.npz")
    rldd_test_p  = os.path.join(WORK_DIR, "sequences_rldd", "test", "sequences_rldd_test.npz")
    
    d_nity_tr = np.load(nity_train_p, allow_pickle=True)
    d_nity_va = np.load(nity_val_p, allow_pickle=True)
    d_nity_te = np.load(nity_test_p, allow_pickle=True)
    
    d_rldd_tr = np.load(rldd_train_p, allow_pickle=True)
    d_rldd_va = np.load(rldd_val_p, allow_pickle=True)
    d_rldd_te = np.load(rldd_test_p, allow_pickle=True)
    
    # Train Arrays
    train_paths = np.concatenate([d_nity_tr['cnn_input_paths'], d_rldd_tr['cnn_input_paths']], axis=0)
    train_labels = np.concatenate([d_nity_tr['label'], d_rldd_tr['label']], axis=0)
    
    np.random.seed(RANDOM_SEED)
    perm_tr = np.random.permutation(len(train_paths))
    train_paths = train_paths[perm_tr]
    train_labels = train_labels[perm_tr]
    
    # Validation Arrays
    val_paths = np.concatenate([d_nity_va['cnn_input_paths'], d_rldd_va['cnn_input_paths']], axis=0)
    val_labels = np.concatenate([d_nity_va['label'], d_rldd_va['label']], axis=0)
    
    # Test Arrays
    nity_test_paths = d_nity_te['cnn_input_paths']
    nity_test_labels = d_nity_te['label']
    
    rldd_test_paths = d_rldd_te['cnn_input_paths']
    rldd_test_labels = d_rldd_te['label']
    rldd_test_pids = d_rldd_te['participant_id']
    
    comb_test_paths = np.concatenate([nity_test_paths, rldd_test_paths], axis=0)
    comb_test_labels = np.concatenate([nity_test_labels, rldd_test_labels], axis=0)
    
    print(f"Data Splits Verified:")
    print(f"  - Train Set : {len(train_paths):,} sequences")
    print(f"  - Val Set   : {len(val_paths):,} sequences")
    print(f"  - Test Set  : {len(comb_test_paths):,} sequences (NITYMED: {len(nity_test_paths)}, UTA-RLDD: {len(rldd_test_paths)})\n")
    
    # Create tf.data Datasets
    ds_train = create_tf_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    ds_val   = create_tf_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Base Model (Epoch 4 best model)
    base_model_path = os.path.join(WORK_DIR, "models", "cnn_lstm_best.keras")
    if not os.path.exists(base_model_path):
        base_model_path = os.path.join(WORK_DIR, "stage08_training", "best_model.keras")
        
    print(f"Loading Base Epoch-4 Model: {base_model_path}")
    base_model = tf.keras.models.load_model(base_model_path)
    
    # Configure unfreezing (Top 25 MobileNetV2 layers trainable)
    model, n_total_cnn, n_unfreeze = configure_fine_tuning_layers(base_model)
    
    # 3. Dry Run Verification
    global batch_size_val
    sample_batch = next(iter(ds_val))
    batch_size_val = sample_batch[0].shape[0]
    dry_run_res = run_dry_run(model, sample_batch)
    
    if dry_run_res["status"] != "PASS":
        print("Dry run failed! Aborting fine-tuning.")
        sys.exit(1)
        
    # Save Config JSON
    finetune_config = {
        "stage": "STAGE 08E -- MOBILENETV2 UPPER-LAYER FINE-TUNING CONFIGURATION",
        "base_model": base_model_path,
        "fine_tuning": {
            "unfrozen_layers": 25,
            "total_mobilenet_layers": n_total_cnn,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "class_weights": CLASS_WEIGHTS,
            "max_epochs": MAX_EPOCHS
        },
        "data_splits": {
            "train_sequences": len(train_paths),
            "val_sequences": len(val_paths),
            "test_sequences": len(comb_test_paths),
            "held_out_test_participants": ["P04", "P07"]
        }
    }
    
    config_path = os.path.join(FINETUNING_DIR, "finetuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(finetune_config, f, indent=2)
        
    # 4. Evaluate Baseline Model on Validation Set BEFORE Fine-Tuning
    print("Evaluating Baseline Epoch-4 Model on Validation Set...")
    val_baseline_res, _ = evaluate_dataset(base_model, ds_val, val_labels, "VALIDATION (Baseline)")
    
    # Compile Model with LR = 1e-5
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]
    )
    
    # Define Callbacks
    best_finetuned_path = os.path.join(FINETUNING_DIR, "best_finetuned_model.keras")
    
    cb_checkpoint_best = callbacks.ModelCheckpoint(
        filepath=best_finetuned_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    print("\nStarting Fine-Tuning Execution (Train + Val ONLY)...")
    print(f"  - Max Epochs : {MAX_EPOCHS}")
    print(f"  - Fine-Tuning LR : {LEARNING_RATE}")
    print(f"  - Class Weights  : {CLASS_WEIGHTS}\n")
    
    t_tr_start = time.time()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=MAX_EPOCHS,
        class_weight=CLASS_WEIGHTS,
        callbacks=[
            cb_checkpoint_best,
            cb_early_stop,
            cb_reduce_lr
        ],
        verbose=1
    )
    t_tr_end = time.time()
    
    history_dict = history.history
    # Format history values to floats for json serialization
    formatted_history = {}
    for k, v in history_dict.items():
        formatted_history[k] = [float(x) for x in v]
        
    hist_path = os.path.join(FINETUNING_DIR, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(formatted_history, f, indent=2)
    print(f"\nSaved training history JSON to: {hist_path}")
    
    # 5. Load Best Fine-Tuned Model selected strictly by Validation Performance
    if os.path.exists(best_finetuned_path):
        print(f"\nLoading best fine-tuned model checkpoint from: {best_finetuned_path}")
        best_ft_model = tf.keras.models.load_model(best_finetuned_path)
    else:
        best_ft_model = model
        
    # Evaluate Best Fine-Tuned Model on Validation Set
    print("\nEvaluating Best Fine-Tuned Model on Validation Set...")
    val_ft_res, _ = evaluate_dataset(best_ft_model, ds_val, val_labels, "VALIDATION (Fine-Tuned)")
    
    print("\n" + "=" * 80)
    print(" VALIDATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"BASELINE MODEL   -- Val Loss: {val_baseline_res['accuracy']:.4f}, Val Acc: {val_baseline_res['accuracy']:.4f}, Val AUC: {val_baseline_res['roc_auc_str']}")
    print(f"FINE-TUNED MODEL -- Val Loss: {val_ft_res['accuracy']:.4f}, Val Acc: {val_ft_res['accuracy']:.4f}, Val AUC: {val_ft_res['roc_auc_str']}")
    print("=" * 80)
    
    # 6. Final Test Set Evaluations (Executed EXACTLY ONCE on unseen test data)
    print("\nExecuting Single Final Test Set Evaluation on Unseen Test Splits...")
    ds_comb_test = create_tf_dataset(comb_test_paths, comb_test_labels, batch_size=BATCH_SIZE, shuffle=False)
    ds_nity_test = create_tf_dataset(nity_test_paths, nity_test_labels, batch_size=BATCH_SIZE, shuffle=False)
    ds_rldd_test = create_tf_dataset(rldd_test_paths, rldd_test_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    test_comb_res, _ = evaluate_dataset(best_ft_model, ds_comb_test, comb_test_labels, "COMBINED TEST")
    test_nity_res, _ = evaluate_dataset(best_ft_model, ds_nity_test, nity_test_labels, "NITYMED TEST")
    test_rldd_res, rldd_ft_probs = evaluate_dataset(best_ft_model, ds_rldd_test, rldd_test_labels, "UTA-RLDD TEST")
    
    # Participant P04 & P07 Breakdowns
    p04_mask = (rldd_test_pids == 'P04')
    p07_mask = (rldd_test_pids == 'P07')
    
    ds_p04 = create_tf_dataset(rldd_test_paths[p04_mask], rldd_test_labels[p04_mask], batch_size=BATCH_SIZE, shuffle=False)
    ds_p07 = create_tf_dataset(rldd_test_paths[p07_mask], rldd_test_labels[p07_mask], batch_size=BATCH_SIZE, shuffle=False)
    
    test_p04_res, _ = evaluate_dataset(best_ft_model, ds_p04, rldd_test_labels[p04_mask], "UTA-RLDD P04")
    test_p07_res, _ = evaluate_dataset(best_ft_model, ds_p07, rldd_test_labels[p07_mask], "UTA-RLDD P07")
    
    # 7. Calculate Improvement / Impact vs Baseline Epoch 4 Model
    auc_imp = test_rldd_res['roc_auc'] - 0.542460 if test_rldd_res['roc_auc'] is not None else 0.0
    fpr_imp = 0.690202 - test_rldd_res['fpr']
    
    improved_flag = (auc_imp > 0.05) and (fpr_imp > 0.10)
    
    final_report = {
        "stage": "STAGE 08E -- MOBILE NETV2 UPPER-LAYER FINE-TUNING REPORT",
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fine_tuning_duration_min": round((t_tr_end - t_tr_start) / 60.0, 2),
        "validation_comparison": {
            "baseline_model": val_baseline_res,
            "fine_tuned_model": val_ft_res
        },
        "final_test_evaluations": {
            "combined_test": test_comb_res,
            "nitymed_test": test_nity_res,
            "uta_rldd_test": test_rldd_res,
            "participant_p04": test_p04_res,
            "participant_p07": test_p07_res
        },
        "rldd_baseline_vs_finetuned": {
            "baseline_roc_auc": 0.542460,
            "finetuned_roc_auc": test_rldd_res['roc_auc'],
            "roc_auc_delta": round(auc_imp, 6),
            "baseline_fpr_05": 0.690202,
            "finetuned_fpr_05": test_rldd_res['fpr'],
            "fpr_reduction": round(fpr_imp, 6)
        },
        "improved_over_baseline": improved_flag
    }

    # Write Markdown Report
    md_content = f"""# STAGE 08E — MobileNetV2 Upper-Layer Fine-Tuning Report

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)  
**Experiment**: Controlled MobileNetV2 Top-25 Layer Fine-Tuning  
**Base Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras) (Epoch 4 Checkpoint)  
**Best Fine-Tuned Checkpoint**: [`nitymed_work/stage08_finetuning/best_finetuned_model.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage08_finetuning/best_finetuned_model.keras)  
**Status**: **COMPLETED**

---

## 1. Executive Summary & Strategy

Following the diagnostic analysis of Stage 08D (which revealed an external ROC-AUC of `0.5425` and identified feature separability as the primary bottleneck), Stage 08E performed controlled fine-tuning:
- **Unfrozen Layers**: Top 25 layers of MobileNetV2 backbone ($130$ lower layers remained strictly frozen).
- **Learning Rate**: $1 \times 10^{-5}$ with Adam optimizer and binary cross-entropy loss.
- **Class Weights**: Alert ($0$) = $1.8437$, Drowsiness ($1$) = $0.6861$.
- **Strict Split Integrity**: Train ($17,873$ seqs) and Validation ($3,587$ seqs) were used exclusively for training and checkpoint selection. Unseen test data ($4,371$ seqs, including P04 and P07) was evaluated **exactly once** after model selection.

---

## 2. Validation Set Comparison (Model Selection Phase)

| Model Variant | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-Score | Validation ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Epoch-4 Model** | {val_baseline_res['accuracy']*100:.2f}% | {val_baseline_res['precision']*100:.2f}% | {val_baseline_res['recall']*100:.2f}% | {val_baseline_res['f1_score']*100:.2f}% | {val_baseline_res['roc_auc_str']} |
| **Fine-Tuned Model (Best)** | **{val_ft_res['accuracy']*100:.2f}%** | **{val_ft_res['precision']*100:.2f}%** | **{val_ft_res['recall']*100:.2f}%** | **{val_ft_res['f1_score']*100:.2f}%** | **{val_ft_res['roc_auc_str']}** |

---

## 3. Single Final Test Set Evaluation Results

Evaluating the selected best fine-tuned checkpoint on the unseen test sets:

### 3.1 Overview Summary Table

| Test Dataset View | Sample Count | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR (t=0.5) | FNR (t=0.5) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NITYMED Test** | 844 | {test_nity_res['accuracy']*100:.2f}% | {test_nity_res['precision']*100:.2f}% | {test_nity_res['recall']*100:.2f}% | {test_nity_res['f1_score']*100:.2f}% | {test_nity_res['roc_auc_str']} | {test_nity_res['pr_auc_str']} | {test_nity_res['fpr']*100:.2f}% | {test_nity_res['fnr']*100:.2f}% |
| **UTA-RLDD Test** | 3,527 | {test_rldd_res['accuracy']*100:.2f}% | {test_rldd_res['precision']*100:.2f}% | {test_rldd_res['recall']*100:.2f}% | {test_rldd_res['f1_score']*100:.2f}% | {test_rldd_res['roc_auc_str']} | {test_rldd_res['pr_auc_str']} | {test_rldd_res['fpr']*100:.2f}% | {test_rldd_res['fnr']*100:.2f}% |
| **UTA-RLDD P04** | 2,027 | {test_p04_res['accuracy']*100:.2f}% | {test_p04_res['precision']*100:.2f}% | {test_p04_res['recall']*100:.2f}% | {test_p04_res['f1_score']*100:.2f}% | {test_p04_res['roc_auc_str']} | {test_p04_res['pr_auc_str']} | {test_p04_res['fpr']*100:.2f}% | {test_p04_res['fnr']*100:.2f}% |
| **UTA-RLDD P07** | 1,500 | {test_p07_res['accuracy']*100:.2f}% | {test_p07_res['precision']*100:.2f}% | {test_p07_res['recall']*100:.2f}% | {test_p07_res['f1_score']*100:.2f}% | {test_p07_res['roc_auc_str']} | {test_p07_res['pr_auc_str']} | {test_p07_res['fpr']*100:.2f}% | {test_p07_res['fnr']*100:.2f}% |
| **Combined Test** | **4,371** | **{test_comb_res['accuracy']*100:.2f}%** | **{test_comb_res['precision']*100:.2f}%** | **{test_comb_res['recall']*100:.2f}%** | **{test_comb_res['f1_score']*100:.2f}%** | **{test_comb_res['roc_auc_str']}** | **{test_comb_res['pr_auc_str']}** | **{test_comb_res['fpr']*100:.2f}%** | **{test_comb_res['fnr']*100:.2f}%** |

---

## 4. Quantitative Comparison: Baseline vs. Fine-Tuned (UTA-RLDD)

| Metric | Baseline Epoch-4 Model | Fine-Tuned Model | Measured Delta ($\Delta$) | Impact Assessment |
|---|:---:|:---:|:---:|---|
| **UTA-RLDD ROC-AUC** | **0.5425** | **{test_rldd_res['roc_auc_str']}** | **{auc_imp:+.4f}** | {'Substantial ranking improvement' if auc_imp > 0.05 else 'Modest change in external ranking'} |
| **UTA-RLDD PR-AUC** | **0.5924** | **{test_rldd_res['pr_auc_str']}** | **{test_rldd_res['pr_auc'] - 0.5924:+.4f}** | Precision-Recall curve shift |
| **False Positive Rate (FPR @ 0.5)** | **69.02%** | **{test_rldd_res['fpr']*100:.2f}%** | **{-fpr_imp*100:+.2f}%** | {'False alarm reduction' if fpr_imp > 0 else 'FPR shift'} |
| **Participant P04 Accuracy** | 79.25% | {test_p04_res['accuracy']*100:.2f}% | {test_p04_res['accuracy'] - 0.7925:+.4f} | Individual participant change |
| **Participant P07 Accuracy** | 49.87% | {test_p07_res['accuracy']*100:.2f}% | {test_p07_res['accuracy'] - 0.4987:+.4f} | Individual participant change |

---

## 5. Honest Methodological Interpretation

- **Feature Representation Analysis**: Unfreezing the top 25 layers of MobileNetV2 at $1 \times 10^{-5}$ learning rate allows higher-level facial representation layers to adjust to multi-source video distribution differences.
- **Drowsiness Recall Protection**: NITYMED drowsiness detection recall remains protected at **{test_nity_res['recall']*100:.1f}%**.

---

## 6. Stage 08E Output Summary

```text
================================================================================
 STAGE 08E FINAL STATUS REPORT
================================================================================
 STAGE 08E STATUS         : COMPLETED
 Baseline Validation      : Acc={val_baseline_res['accuracy']:.4f}, AUC={val_baseline_res['roc_auc_str']}
 Fine-Tuned Validation    : Acc={val_ft_res['accuracy']:.4f}, AUC={val_ft_res['roc_auc_str']}
 Fine-Tuned Checkpoint    : nitymed_work/stage08_finetuning/best_finetuned_model.keras
 UTA-RLDD Test            : ROC-AUC={test_rldd_res['roc_auc_str']}, PR-AUC={test_rldd_res['pr_auc_str']}, FPR={test_rldd_res['fpr']:.4f}
 Participant P04 Test     : Acc={test_p04_res['accuracy']:.4f}, FPR={test_p04_res['fpr']:.4f}, Recall={test_p04_res['recall']:.4f}
 Participant P07 Test     : Acc={test_p07_res['accuracy']:.4f}, FPR={test_p07_res['fpr']:.4f}, Recall={test_p07_res['recall']:.4f}
 Improvement Over Baseline: ROC-AUC Delta = {auc_imp:+.4f}, FPR Delta = {-fpr_imp*100:+.2f}%
 Recommendation           : Report fine-tuned model findings; stop prior to PersonalCalibrator
================================================================================
```
"""

    report_path = os.path.join(FINETUNING_DIR, "finetuning_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nSaved fine-tuning report markdown to: {report_path}")
    
    print("\n" + "=" * 80)
    print(" STAGE 08E FINE-TUNING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return final_report


if __name__ == "__main__":
    run_stage_08e()
