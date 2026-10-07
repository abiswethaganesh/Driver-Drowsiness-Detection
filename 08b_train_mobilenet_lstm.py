r"""
08b_train_mobilenet_lstm.py
STAGE 08B -- MobileNetV2 + LSTM Training Pipeline & Multi-Dataset Evaluation

Trains the MobileNetV2 + LSTM temporal drowsiness classifier on the combined
NITYMED + UTA-RLDD (Fold 1) dataset.

Pipeline details:
- Input shape: (16, 128, 128, 3)
- Backbone: MobileNetV2 (ImageNet weights, Frozen)
- Temporal modeling: LSTM(64) + Dropout(0.3) + Dense(32, relu) + Dropout(0.2) + Dense(1, sigmoid)
- Optimizer: Adam (lr = 1e-4)
- Loss: Binary cross-entropy with class weights {0: 1.8437, 1: 0.6861}
- Batch size: 8
- Max epochs: 25
- Streaming data pipeline (tf.data) reading crop JPEG paths on demand
- Per-epoch checkpointing for safe resumption
- Separate test evaluation: Combined, NITYMED, and UTA-RLDD test sets

Saved to: WORK_DIR/stage08_training/
"""

import os
import sys
import json
import time
import glob
import psutil
import numpy as np
import pandas as pd
from importlib import import_module

# Force stdout encoding to utf-8 if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load CFG
CFG = import_module("00_config").CFG
WORK_DIR = CFG.WORK_DIR
STAGE08_DIR = os.path.join(WORK_DIR, "stage08_training")
CHECKPOINTS_DIR = os.path.join(STAGE08_DIR, "checkpoints")
os.makedirs(STAGE08_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, callbacks
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 25
LEARNING_RATE = 1e-4
CLASS_WEIGHTS = {0: 1.8437, 1: 0.6861}


def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def build_mobilenet_lstm_model(input_shape=(16, 128, 128, 3)):
    """
    Constructs MobileNetV2 + LSTM model. MobileNetV2 backbone is FROZEN.
    """
    cnn_base = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    cnn_base.trainable = False  # Freeze MobileNetV2 backbone
    
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = layers.TimeDistributed(cnn_base, name="time_distributed_mobilenet")(inputs)
    x = layers.LSTM(64, return_sequences=False, name="lstm_temporal")(x)
    x = layers.Dropout(0.3, name="dropout_lstm")(x)
    x = layers.Dense(32, activation="relu", name="dense_head")(x)
    x = layers.Dropout(0.2, name="dropout_dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="drowsiness_score")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_LSTM_Drowsiness_Classifier")
    return model


def load_and_preprocess_image(path_tensor):
    img_raw = tf.io.read_file(path_tensor)
    img_dec = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize(img_dec, [128, 128])
    img_norm = (img_resized / 127.5) - 1.0  # [-1, 1]
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


class EpochLoggerCallback(callbacks.Callback):
    def __init__(self, timing_file, history_file):
        super().__init__()
        self.timing_file = timing_file
        self.history_file = history_file
        self.epoch_timings = []
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        logs = logs or {}
        entry = {
            "epoch": epoch + 1,
            "duration_sec": round(elapsed, 2),
            "loss": round(float(logs.get("loss", 0)), 6),
            "accuracy": round(float(logs.get("accuracy", 0)), 6),
            "precision": round(float(logs.get("precision", 0)), 6),
            "recall": round(float(logs.get("recall", 0)), 6),
            "auc": round(float(logs.get("auc", 0)), 6),
            "val_loss": round(float(logs.get("val_loss", 0)), 6),
            "val_accuracy": round(float(logs.get("val_accuracy", 0)), 6),
            "val_precision": round(float(logs.get("val_precision", 0)), 6),
            "val_recall": round(float(logs.get("val_recall", 0)), 6),
            "val_auc": round(float(logs.get("val_auc", 0)), 6),
            "learning_rate": round(float(self.model.optimizer.learning_rate.numpy()), 6)
        }
        self.epoch_timings.append(entry)
        
        with open(self.timing_file, "w", encoding="utf-8") as f:
            json.dump(self.epoch_timings, f, indent=2)
            
        df_hist = pd.DataFrame(self.epoch_timings)
        df_hist.to_csv(self.history_file, index=False)
        
        print(f" -> Epoch {epoch+1:02d}/{MAX_EPOCHS:02d} completed in {elapsed:.2f}s | "
              f"loss: {entry['loss']:.4f} - acc: {entry['accuracy']:.4f} - auc: {entry['auc']:.4f} | "
              f"val_loss: {entry['val_loss']:.4f} - val_acc: {entry['val_accuracy']:.4f} - val_auc: {entry['val_auc']:.4f}")


def evaluate_model_on_dataset(model, dataset, true_labels, name):
    print(f"\nEvaluating model on {name} ({len(true_labels):,} sequences)...")
    t0 = time.time()
    preds_prob = model.predict(dataset, verbose=0).flatten()
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
    
    res = {
        "dataset_name": name,
        "sample_count": len(true_labels),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "auc": round(float(auc), 6) if auc is not None else None,
        "auc_str": auc_str,
        "confusion_matrix": {
            "TN": cm[0][0], "FP": cm[0][1],
            "FN": cm[1][0], "TP": cm[1][1]
        },
        "inference_time_sec": round(t1 - t0, 2),
        "ms_per_sequence": round((t1 - t0) / len(true_labels) * 1000, 2)
    }
    return res


def run_stage_08b_training():
    print("=" * 80)
    print(" STAGE 08B -- MOBILENETV2 + LSTM FULL TRAINING RUN")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Device Verification
    gpus = tf.config.list_physical_devices('GPU')
    device_name = "GPU" if gpus else "CPU"
    print(f"Execution Device: {device_name} ({len(gpus)} GPU(s) detected)")
    print(f"Current Process RAM: {get_ram_usage_mb():.2f} MB\n")
    
    # 2. Load NPZ Split Paths & Labels
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
    
    # Combined Training Arrays
    train_paths = np.concatenate([d_nity_tr['cnn_input_paths'], d_rldd_tr['cnn_input_paths']], axis=0)
    train_labels = np.concatenate([d_nity_tr['label'], d_rldd_tr['label']], axis=0)
    
    # Shuffle Train set reproducibly with seed 42
    np.random.seed(RANDOM_SEED)
    perm_tr = np.random.permutation(len(train_paths))
    train_paths = train_paths[perm_tr]
    train_labels = train_labels[perm_tr]
    
    # Combined Validation Arrays
    val_paths = np.concatenate([d_nity_va['cnn_input_paths'], d_rldd_va['cnn_input_paths']], axis=0)
    val_labels = np.concatenate([d_nity_va['label'], d_rldd_va['label']], axis=0)
    
    # Test Subsets
    nity_test_paths = d_nity_te['cnn_input_paths']
    nity_test_labels = d_nity_te['label']
    
    rldd_test_paths = d_rldd_te['cnn_input_paths']
    rldd_test_labels = d_rldd_te['label']
    
    comb_test_paths = np.concatenate([nity_test_paths, rldd_test_paths], axis=0)
    comb_test_labels = np.concatenate([nity_test_labels, rldd_test_labels], axis=0)
    
    print("Split Summaries:")
    print(f"  - Train Set : {len(train_paths):,} sequences (NITYMED: {len(d_nity_tr['label']):,}, UTA-RLDD: {len(d_rldd_tr['label']):,})")
    print(f"  - Val Set   : {len(val_paths):,} sequences (NITYMED: {len(d_nity_va['label']):,}, UTA-RLDD: {len(d_rldd_va['label']):,})")
    print(f"  - Test Set  : {len(comb_test_paths):,} sequences (NITYMED: {len(nity_test_paths):,}, UTA-RLDD: {len(rldd_test_paths):,})")

    # 3. Create tf.data.Datasets
    print("\nCreating streaming tf.data Datasets...")
    ds_train = create_tf_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    ds_val   = create_tf_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    ds_comb_test = create_tf_dataset(comb_test_paths, comb_test_labels, batch_size=BATCH_SIZE, shuffle=False)
    ds_nity_test = create_tf_dataset(nity_test_paths, nity_test_labels, batch_size=BATCH_SIZE, shuffle=False)
    ds_rldd_test = create_tf_dataset(rldd_test_paths, rldd_test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Build Model
    print("\nBuilding MobileNetV2 + LSTM Model...")
    model = build_mobilenet_lstm_model(input_shape=(16, 128, 128, 3))
    
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

    # 5. Define Callbacks
    best_model_path = os.path.join(STAGE08_DIR, "best_model.keras")
    final_model_path = os.path.join(STAGE08_DIR, "final_model.keras")
    timing_file = os.path.join(STAGE08_DIR, "epoch_timing.json")
    history_csv = os.path.join(STAGE08_DIR, "training_history.csv")
    
    checkpoint_format = os.path.join(CHECKPOINTS_DIR, "epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras")
    
    cb_checkpoint_best = callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    
    cb_checkpoint_epoch = callbacks.ModelCheckpoint(
        filepath=checkpoint_format,
        save_weights_only=False,
        verbose=0
    )
    
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    cb_logger = EpochLoggerCallback(timing_file=timing_file, history_file=history_csv)

    print("\nStarting Training Run...")
    print(f"  - Maximum Epochs : {MAX_EPOCHS}")
    print(f"  - Batch Size     : {BATCH_SIZE}")
    print(f"  - Class Weights  : {CLASS_WEIGHTS}")
    print(f"  - Initial LR     : {LEARNING_RATE}\n")

    # Check if existing epoch timing log exists to track resume
    initial_epoch = 0
    if os.path.exists(timing_file):
        try:
            with open(timing_file, "r", encoding="utf-8") as f:
                past_timings = json.load(f)
            if past_timings:
                initial_epoch = len(past_timings)
                cb_logger.epoch_timings = past_timings
                print(f"Detected {initial_epoch} completed epochs in timing log.")
        except Exception:
            initial_epoch = 0

    # Load latest checkpoint if resuming
    if initial_epoch > 0:
        ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "epoch_*.keras")))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f"Loading model/weights from latest checkpoint: {latest_ckpt}")
            try:
                model = models.load_model(latest_ckpt)
            except Exception as e:
                print(f"Loading weights into model from {latest_ckpt}: {e}")
                model.load_weights(latest_ckpt)
        else:
            print("No checkpoint files found in directory, keeping initial model.")

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=MAX_EPOCHS,
        initial_epoch=initial_epoch,
        class_weight=CLASS_WEIGHTS,
        callbacks=[
            cb_checkpoint_best,
            cb_checkpoint_epoch,
            cb_early_stop,
            cb_reduce_lr,
            cb_logger
        ],
        verbose=1
    )

    total_training_time_sec = time.time() - start_time
    
    # Save Final Model
    model.save(final_model_path)
    print(f"\nSaved final model to: {final_model_path}")
    
    # Load Best Model for Evaluation
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint model from: {best_model_path}")
        eval_model = models.load_model(best_model_path)
    else:
        eval_model = model

    # 6. Separate Test Evaluations
    print("\nRunning separate dataset-source test evaluations...")
    res_comb = evaluate_model_on_dataset(eval_model, ds_comb_test, comb_test_labels, "COMBINED TEST")
    res_nity = evaluate_model_on_dataset(eval_model, ds_nity_test, nity_test_labels, "NITYMED TEST")
    res_rldd = evaluate_model_on_dataset(eval_model, ds_rldd_test, rldd_test_labels, "UTA-RLDD TEST")

    # 7. Aggregate & Save Evaluation Results
    history_data = cb_logger.epoch_timings
    best_epoch_entry = min(history_data, key=lambda x: x["val_loss"]) if history_data else {}
    best_epoch_idx = best_epoch_entry.get("epoch", 0)
    best_val_loss = best_epoch_entry.get("val_loss", 0.0)
    best_val_auc = best_epoch_entry.get("val_auc", 0.0)

    eval_summary = {
        "status": "COMPLETED",
        "epochs_completed": len(history_data),
        "best_epoch": best_epoch_idx,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "total_training_time_min": round(total_training_time_sec / 60.0, 2),
        "avg_epoch_time_sec": round(np.mean([e["duration_sec"] for e in history_data]), 2) if history_data else 0.0,
        "combined_test": res_comb,
        "nitymed_test": res_nity,
        "uta_rldd_test": res_rldd
    }

    eval_file = os.path.join(STAGE08_DIR, "evaluation_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nSaved complete evaluation results JSON to: {eval_file}")

    # 8. Print Final Report
    print("\n" + "=" * 80)
    print(" STAGE 08B TRAINING & EVALUATION FINAL REPORT")
    print("=" * 80)
    print("STAGE 08B TRAINING STATUS: COMPLETED\n")
    
    print(f"Epochs completed: {len(history_data)}")
    print(f"Best epoch: {best_epoch_idx}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation AUC: {best_val_auc:.6f}")
    
    if history_data:
        last_e = history_data[-1]
        print(f"\nFinal training metrics (Epoch {last_e['epoch']}):")
        print(f"  - Loss: {last_e['loss']:.6f}, Acc: {last_e['accuracy']:.4f}, Prec: {last_e['precision']:.4f}, Rec: {last_e['recall']:.4f}, AUC: {last_e['auc']:.4f}")
        print(f"Final validation metrics (Epoch {last_e['epoch']}):")
        print(f"  - Val Loss: {last_e['val_loss']:.6f}, Val Acc: {last_e['val_accuracy']:.4f}, Val Prec: {last_e['val_precision']:.4f}, Val Rec: {last_e['val_recall']:.4f}, Val AUC: {last_e['val_auc']:.4f}")

    print(f"\nTraining time: {total_training_time_sec/60.0:.2f} minutes")
    print(f"Average epoch time: {eval_summary['avg_epoch_time_sec']:.2f} seconds")
    
    print("\n" + "=" * 80)
    print(" TEST RESULTS SUMMARY BY DATASET SOURCE")
    print("=" * 80)
    
    for r in [res_comb, res_nity, res_rldd]:
        print(f"\n{r['dataset_name']} ({r['sample_count']:,} samples):")
        print(f"  Accuracy       : {r['accuracy']:.6f}")
        print(f"  Precision      : {r['precision']:.6f}")
        print(f"  Recall         : {r['recall']:.6f}")
        print(f"  F1-Score       : {r['f1_score']:.6f}")
        print(f"  AUC            : {r['auc_str']}")
        print(f"  Confusion Matrix (TN, FP, FN, TP):")
        print(f"    [[{r['confusion_matrix']['TN']:5d}, {r['confusion_matrix']['FP']:5d}],")
        print(f"     [{r['confusion_matrix']['FN']:5d}, {r['confusion_matrix']['TP']:5d}]]")
        
    print("\n" + "=" * 80)
    print(" IMPORTANT RESEARCH DISCLAIMER")
    print("=" * 80)
    print(" This model is a research prototype temporal drowsiness classifier.")
    print(" It is NOT production-ready and MUST NOT be used for real vehicle control.")
    print(" Do NOT proceed to intervention/recovery module without authorization.")
    print("=" * 80)


if __name__ == "__main__":
    run_stage_08b_training()
