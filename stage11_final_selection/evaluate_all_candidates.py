"""
STAGE 11 — ALL CANDIDATE MODEL EVALUATOR
Evaluates all existing candidate models across Validation, Test splits, and Live Webcam simulations.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

BASE_DIR = r"d:\Sem 7\NNDL\Project Demo"
STAGE11_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage11_final_selection")
os.makedirs(STAGE11_DIR, exist_ok=True)

# Define Candidate Models to Evaluate
CANDIDATE_MODELS = [
    {
        "Experiment": "Stage 08 Baseline",
        "Model_Path": os.path.join(BASE_DIR, r"nitymed_work\models\cnn_lstm_best.keras"),
        "Preprocessing": "MobileNetV2 A1 (x/127.5 - 1.0)",
        "Architecture": "MobileNetV2 + LSTM(64)",
        "Trainable_MobileNet_Layers": 0,
        "norm_type": "A1"
    },
    {
        "Experiment": "Stage 08E Fine-Tuning",
        "Model_Path": os.path.join(BASE_DIR, r"nitymed_work\stage08_finetuning\best_finetuned_model.keras"),
        "Preprocessing": "MobileNetV2 A1 (x/127.5 - 1.0)",
        "Architecture": "MobileNetV2 (Top 25) + LSTM(64)",
        "Trainable_MobileNet_Layers": 25,
        "norm_type": "A1"
    },
    {
        "Experiment": "Stage 10B Domain-Aware",
        "Model_Path": os.path.join(BASE_DIR, r"nitymed_work\stage10b_domain_training\best_domain_model.keras"),
        "Preprocessing": "MobileNetV2 A1 (x/127.5 - 1.0)",
        "Architecture": "MobileNetV2 + LSTM(64)",
        "Trainable_MobileNet_Layers": 0,
        "norm_type": "A1"
    },
    {
        "Experiment": "Stage 10C Candidate (Model B2)",
        "Model_Path": os.path.join(BASE_DIR, r"nitymed_work\stage10c_domain_adaptation\model_b2_temporal_avg.keras"),
        "Preprocessing": "Per-Sequence Luminance Norm (A3)",
        "Architecture": "MobileNetV2 + GlobalAvgPool1D",
        "Trainable_MobileNet_Layers": 0,
        "norm_type": "A3"
    }
]

# Load Datasets
print("--- Loading Validation and Test Datasets ---")
nm_val = np.load(os.path.join(BASE_DIR, r"nitymed_work\sequences\val\sequences_val.npz"), allow_pickle=True)
nm_test = np.load(os.path.join(BASE_DIR, r"nitymed_work\sequences\test\sequences_test.npz"), allow_pickle=True)

rldd_val = np.load(os.path.join(BASE_DIR, r"nitymed_work\sequences_rldd\val\sequences_rldd_val.npz"), allow_pickle=True)
rldd_test = np.load(os.path.join(BASE_DIR, r"nitymed_work\sequences_rldd\test\sequences_rldd_test.npz"), allow_pickle=True)

# Validation set
val_paths = np.concatenate([nm_val['cnn_input_paths'], rldd_val['cnn_input_paths']], axis=0)
val_labels = np.concatenate([np.ones(len(nm_val['cnn_input_paths']), dtype=np.int32), rldd_val['label'].astype(np.int32)], axis=0)

# Test sets
nm_test_paths = nm_test['cnn_input_paths']
nm_test_labels = np.ones(len(nm_test_paths), dtype=np.int32)

rldd_test_paths = rldd_test['cnn_input_paths']
rldd_test_labels = rldd_test['label'].astype(np.int32)
rldd_test_participants = rldd_test['participant_id']

test_paths_combined = np.concatenate([nm_test_paths, rldd_test_paths], axis=0)
test_labels_combined = np.concatenate([nm_test_labels, rldd_test_labels], axis=0)

p04_mask = (rldd_test_participants == 'P04')
p07_mask = (rldd_test_participants == 'P07')

p04_paths, p04_labels = rldd_test_paths[p04_mask], rldd_test_labels[p04_mask]
p07_paths, p07_labels = rldd_test_paths[p07_mask], rldd_test_labels[p07_mask]

# Preprocessing helpers
def apply_norm_a1(img_float):
    return (img_float / 127.5) - 1.0

def apply_norm_a3_seq(seq_float):
    lum = 0.299 * seq_float[:, :, :, 0] + 0.587 * seq_float[:, :, :, 1] + 0.114 * seq_float[:, :, :, 2]
    mean = tf.reduce_mean(lum)
    std = tf.maximum(tf.math.reduce_std(lum), 1.0)
    adjusted = ((seq_float - mean) / std) * 50.0 + 128.0
    adjusted = tf.clip_by_value(adjusted, 0.0, 255.0)
    return (adjusted / 127.5) - 1.0

def load_and_preprocess_frame(path_tensor, norm_type="A1"):
    img_raw = tf.io.read_file(path_tensor)
    img_dec = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.cast(tf.image.resize(img_dec, [128, 128]), tf.float32)
    if norm_type == "A1":
        return apply_norm_a1(img_resized)
    else:
        return img_resized

def parse_sequence(paths_tensor, label_tensor, norm_type="A1"):
    if norm_type == "A3":
        raw_frames = tf.map_fn(
            lambda p: load_and_preprocess_frame(p, norm_type="A3"),
            paths_tensor,
            fn_output_signature=tf.float32,
            parallel_iterations=16
        )
        seq_frames = apply_norm_a3_seq(raw_frames)
    else:
        seq_frames = tf.map_fn(
            lambda p: load_and_preprocess_frame(p, norm_type="A1"),
            paths_tensor,
            fn_output_signature=tf.float32,
            parallel_iterations=16
        )
    return seq_frames, label_tensor

def create_tf_dataset(paths_array, labels_array, norm_type="A1", batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((paths_array, labels_array))
    ds = ds.map(lambda p, l: parse_sequence(p, l, norm_type=norm_type), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def evaluate_model_split(model, paths, labels, norm_type="A1"):
    if len(paths) == 0:
        return {}
    ds = create_tf_dataset(paths, labels, norm_type=norm_type, batch_size=32)
    probs = model.predict(ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    
    acc = float(accuracy_score(labels, preds))
    unique_y = np.unique(labels)
    if len(unique_y) > 1:
        prec = float(precision_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        f1 = float(f1_score(labels, preds, zero_division=0))
        roc_auc = float(roc_auc_score(labels, probs))
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        prec = float(precision_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        f1 = float(f1_score(labels, preds, zero_division=0))
        roc_auc = None
        fpr = None
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "fpr": fpr, "mean_prob": float(np.mean(probs)), "probs": probs
    }

def simulate_live_webcam(model, norm_type="A1"):
    # Generate 100 sequences of neutral/alert simulation tensor frames
    np.random.seed(42)
    sim_probs = []
    # Create 50 synthetic frames of varying lighting
    for _ in range(50):
        mean_lum = np.random.randint(80, 180)
        frame_raw = np.clip(np.random.normal(mean_lum, 25, (16, 128, 128, 3)), 0, 255).astype(np.float32)
        if norm_type == "A1":
            frame_norm = (frame_raw / 127.5) - 1.0
        elif norm_type == "A3":
            # Per sequence luminance norm
            seq_tf = tf.convert_to_tensor(frame_raw, dtype=tf.float32)
            frame_norm = apply_norm_a3_seq(seq_tf).numpy()
        else:
            frame_norm = (frame_raw / 127.5) - 1.0
        
        batch = np.expand_dims(frame_norm, axis=0)
        p = float(model.predict(batch, verbose=0)[0][0])
        sim_probs.append(p)
    
    mean_p = float(np.mean(sim_probs))
    med_p = float(np.median(sim_probs))
    sat_rate = float(np.mean(np.array(sim_probs) >= 0.90))
    return mean_p, med_p, sat_rate

# Run Evaluation across candidates
results = []
print("\n" + "="*60)
print("STARTING CANDIDATE EVALUATION")
print("="*60)

for cand in CANDIDATE_MODELS:
    name = cand["Experiment"]
    m_path = cand["Model_Path"]
    norm = cand["norm_type"]
    
    if not os.path.exists(m_path):
        print(f"Skipping {name}: Model file not found at {m_path}")
        continue
        
    print(f"\nEvaluating candidate: {name} ({m_path})...")
    model = tf.keras.models.load_model(m_path)
    
    # 1. Validation
    val_res = evaluate_model_split(model, val_paths, val_labels, norm_type=norm)
    print(f"  Val ROC-AUC: {val_res.get('roc_auc')}, Val F1: {val_res.get('f1'):.4f}")
    
    # 2. UTA Test
    uta_res = evaluate_model_split(model, rldd_test_paths, rldd_test_labels, norm_type=norm)
    print(f"  UTA Test ROC-AUC: {uta_res.get('roc_auc')}, UTA F1: {uta_res.get('f1'):.4f}, FPR: {uta_res.get('fpr')}")
    
    # 3. Combined Test
    comb_res = evaluate_model_split(model, test_paths_combined, test_labels_combined, norm_type=norm)
    print(f"  Combined Test ROC-AUC: {comb_res.get('roc_auc')}, Combined F1: {comb_res.get('f1'):.4f}")
    
    # 4. NITYMED Test
    nm_res = evaluate_model_split(model, nm_test_paths, nm_test_labels, norm_type=norm)
    print(f"  NITYMED Recall: {nm_res.get('recall')}")
    
    # 5. P04
    p04_res = evaluate_model_split(model, p04_paths, p04_labels, norm_type=norm)
    print(f"  P04 ROC-AUC: {p04_res.get('roc_auc')}")
    
    # 6. P07
    p07_res = evaluate_model_split(model, p07_paths, p07_labels, norm_type=norm)
    print(f"  P07 ROC-AUC: {p07_res.get('roc_auc')}")
    
    # 7. Live simulation
    live_mean, live_med, live_sat = simulate_live_webcam(model, norm_type=norm)
    print(f"  Live Mean: {live_mean:.4f}, Live Med: {live_med:.4f}, Saturation: {live_sat*100:.1f}%")
    
    results.append({
        "Experiment": cand["Experiment"],
        "Model_Path": cand["Model_Path"],
        "Preprocessing": cand["Preprocessing"],
        "Architecture": cand["Architecture"],
        "Trainable_MobileNet_Layers": cand["Trainable_MobileNet_Layers"],
        "Validation_ROC_AUC": round(val_res.get("roc_auc"), 6) if val_res.get("roc_auc") is not None else "N/A",
        "Validation_F1": round(val_res.get("f1"), 6) if val_res.get("f1") is not None else "N/A",
        "UTA_Test_ROC_AUC": round(uta_res.get("roc_auc"), 6) if uta_res.get("roc_auc") is not None else "N/A",
        "UTA_Test_F1": round(uta_res.get("f1"), 6) if uta_res.get("f1") is not None else "N/A",
        "UTA_Test_FPR": round(uta_res.get("fpr"), 6) if uta_res.get("fpr") is not None else "N/A",
        "Combined_Test_ROC_AUC": round(comb_res.get("roc_auc"), 6) if comb_res.get("roc_auc") is not None else "N/A",
        "Combined_Test_F1": round(comb_res.get("f1"), 6) if comb_res.get("f1") is not None else "N/A",
        "NITYMED_Test_Recall": round(nm_res.get("recall"), 6) if nm_res.get("recall") is not None else "N/A",
        "P04_ROC_AUC": round(p04_res.get("roc_auc"), 6) if p04_res.get("roc_auc") is not None else "N/A",
        "P07_ROC_AUC": round(p07_res.get("roc_auc"), 6) if p07_res.get("roc_auc") is not None else "N/A",
        "Live_Mean_Probability": round(live_mean, 6),
        "Live_Median_Probability": round(live_med, 6),
        "Live_Saturation_Rate": round(live_sat, 6)
    })

df_comp = pd.DataFrame(results)
comp_csv_path = os.path.join(STAGE11_DIR, "master_comparison.csv")
df_comp.to_csv(comp_csv_path, index=False)
print(f"\nSaved Master Comparison Table to: {comp_csv_path}")

# Also save detailed json
with open(os.path.join(STAGE11_DIR, "candidate_evaluations.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
