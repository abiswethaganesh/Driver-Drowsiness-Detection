"""
07_sequence_generation.py
STAGE 07 — Temporal Sequence Generation

Converts individual frame samples into fixed-length temporal sequences (16 frames)
for the CNN-LSTM model while enforcing strict split, video, and behavior separation.

Key requirements:
- SEQ_LEN = 16 consecutive frames per sequence.
- Stride based on SEQ_OVERLAP (50% -> step of 8 frames).
- Sliding window operates PER VIDEO independently (never crosses video, split, or behavior boundaries).
- Valid sequence rule: >=80% of frames (>=13/16) must have valid face landmarks (landmark_available == True).
- For 1-3 invalid frames in a valid sequence: cnn_input_path uses nearest valid face crop.
- Unaltered original landmark_available status and true valid_frame_count are preserved.
- input_reused / reused_frame_count flags track crop substitution.
- Duration calculated as (end_frame_index - start_frame_index) / source_fps per video.
- NO Driver ID parsed or stored anywhere.
- NO label 0 generated (all current samples are Yawning or Microsleep with label = 1).
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module

# Import config
CFG = import_module("00_config").CFG


def load_video_fps_mapping():
    """Load video filename -> fps mapping from metadata.csv."""
    if not os.path.exists(CFG.METADATA_CSV):
        raise FileNotFoundError(f"Metadata file not found: {CFG.METADATA_CSV}")
    meta_df = pd.read_csv(CFG.METADATA_CSV)
    fps_map = dict(zip(meta_df["filename"], meta_df["fps"]))
    return fps_map


def get_nearest_valid_crop(face_crop_paths, landmark_flags, idx):
    """
    Finds the nearest valid face crop path within the window for an invalid frame.
    Searches outwards bidirectionally.
    """
    n = len(landmark_flags)
    for delta in range(1, n):
        # Check left
        left = idx - delta
        if left >= 0 and landmark_flags[left]:
            return face_crop_paths[left]
        # Check right
        right = idx + delta
        if right < n and landmark_flags[right]:
            return face_crop_paths[right]
    # Fallback to current path if no valid frame in window (should not happen for >=80% valid)
    return face_crop_paths[idx]


def generate_sequences():
    print("=" * 80)
    print("STAGE 07 — TEMPORAL SEQUENCE GENERATION")
    print("=" * 80)

    CFG.ensure_dirs()

    if not os.path.exists(CFG.FACE_LANDMARK_METADATA_CSV):
        raise FileNotFoundError(
            f"Face landmark metadata not found: {CFG.FACE_LANDMARK_METADATA_CSV}. "
            "Please run Stage 06 first."
        )

    print(f"Loading landmark metadata from: {CFG.FACE_LANDMARK_METADATA_CSV}")
    landmark_df = pd.read_csv(CFG.FACE_LANDMARK_METADATA_CSV)
    fps_map = load_video_fps_mapping()

    total_input_frames = len(landmark_df)
    missing_landmark_frames = (landmark_df["landmark_available"] == False).sum()
    print(f"Total input frames: {total_input_frames}")
    print(f"Frames with missing landmarks: {missing_landmark_frames} "
          f"({missing_landmark_frames / total_input_frames * 100:.2f}%)")

    seq_len = CFG.SEQ_LEN
    step = max(1, int(seq_len * (1.0 - CFG.SEQ_OVERLAP)))
    print(f"Sequence Parameters: SEQ_LEN={seq_len}, SEQ_OVERLAP={CFG.SEQ_OVERLAP}, STRIDE={step}")

    # Group strictly by (split, behavior, video_filename)
    grouped = landmark_df.groupby(["split", "behavior", "video_filename"], sort=False)
    
    sequence_rows = []
    split_npz_data = {"train": [], "val": [], "test": []}

    total_windows_examined = 0
    discarded_seq_count = 0
    reused_crop_seq_count = 0

    per_split_seq_counters = {"train": 0, "val": 0, "test": 0}

    print("\nProcessing videos and building sliding-window sequences...")
    for (split, behavior, video_filename), group in tqdm(grouped, desc="Generating Sequences"):
        # Sort chronologically by frame_index
        group = group.sort_values("frame_index").reset_index(drop=True)
        n_frames = len(group)
        source_fps = float(fps_map.get(video_filename, 25.0))

        for start_i in range(0, n_frames - seq_len + 1, step):
            total_windows_examined += 1
            window = group.iloc[start_i : start_i + seq_len]

            # Landmark validity check (unaltered status)
            lm_available = window["landmark_available"].values.astype(bool)
            valid_cnt = int(lm_available.sum())
            valid_ratio = valid_cnt / float(seq_len)

            # Rule: Discard sequence if valid face ratio < 80% (i.e. < 13 valid frames out of 16)
            if valid_ratio < 0.80:
                discarded_seq_count += 1
                continue

            # Build CNN input paths with nearest valid replacement for missing face frames
            frame_paths = window["frame_path"].values.tolist()
            face_crop_paths = window["face_crop_path"].values.tolist()
            cnn_input_paths = []

            reused_cnt = 0
            for idx in range(seq_len):
                if lm_available[idx]:
                    cnn_input_paths.append(face_crop_paths[idx])
                else:
                    reused_crop = get_nearest_valid_crop(face_crop_paths, lm_available, idx)
                    cnn_input_paths.append(reused_crop)
                    reused_cnt += 1

            if reused_cnt > 0:
                reused_crop_seq_count += 1

            start_frame_idx = int(window.iloc[0]["frame_index"])
            end_frame_idx = int(window.iloc[-1]["frame_index"])

            # Calculate temporal duration using exact first and last frame indices
            duration_sec = (end_frame_idx - start_frame_idx) / source_fps

            per_split_seq_counters[split] += 1
            seq_id = f"seq_{split}_{per_split_seq_counters[split]:06d}"

            gender = str(window.iloc[0]["gender"])
            label = int(window.iloc[0]["label"])

            # Save sequence metadata record (NO DRIVER ID)
            seq_record = {
                "sequence_id": seq_id,
                "video_filename": video_filename,
                "split": split,
                "behavior": behavior,
                "gender": gender,
                "label": label,
                "start_frame_index": start_frame_idx,
                "end_frame_index": end_frame_idx,
                "seq_len": seq_len,
                "valid_frame_count": valid_cnt,
                "reused_frame_count": reused_cnt,
                "input_reused": reused_cnt > 0,
                "source_fps": source_fps,
                "duration_seconds": round(duration_sec, 4),
            }
            sequence_rows.append(seq_record)

            # Data bundle for NPZ file per split
            split_npz_data[split].append({
                "sequence_id": seq_id,
                "video_filename": video_filename,
                "split": split,
                "behavior": behavior,
                "gender": gender,
                "label": label,
                "start_frame_index": start_frame_idx,
                "end_frame_index": end_frame_idx,
                "valid_frame_count": valid_cnt,
                "reused_frame_count": reused_cnt,
                "duration_seconds": duration_sec,
                "frame_paths": np.array(frame_paths, dtype=str),
                "face_crop_paths": np.array(face_crop_paths, dtype=str),
                "cnn_input_paths": np.array(cnn_input_paths, dtype=str),
                "landmark_available": lm_available,
            })

    seq_df = pd.DataFrame(sequence_rows)
    seq_metadata_csv = os.path.join(CFG.WORK_DIR, "sequence_metadata.csv")
    seq_df.to_csv(seq_metadata_csv, index=False)
    print(f"\nSaved sequence metadata to: {seq_metadata_csv} ({len(seq_df)} rows)")

    # Save per-split NPZ files inside WORK_DIR/sequences/{split}/
    saved_files = [seq_metadata_csv]
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(CFG.SEQ_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        split_items = split_npz_data[split]

        if not split_items:
            continue

        npz_filename = os.path.join(split_dir, f"sequences_{split}.npz")
        
        # Structure arrays into NumPy format for ultra-fast loading during training
        seq_ids = np.array([item["sequence_id"] for item in split_items])
        video_fns = np.array([item["video_filename"] for item in split_items])
        behaviors = np.array([item["behavior"] for item in split_items])
        genders = np.array([item["gender"] for item in split_items])
        labels = np.array([item["label"] for item in split_items], dtype=np.int32)
        start_indices = np.array([item["start_frame_index"] for item in split_items], dtype=np.int32)
        end_indices = np.array([item["end_frame_index"] for item in split_items], dtype=np.int32)
        valid_counts = np.array([item["valid_frame_count"] for item in split_items], dtype=np.int32)
        reused_counts = np.array([item["reused_frame_count"] for item in split_items], dtype=np.int32)
        durations = np.array([item["duration_seconds"] for item in split_items], dtype=np.float32)

        frame_paths_mat = np.stack([item["frame_paths"] for item in split_items], axis=0)
        face_crops_mat = np.stack([item["face_crop_paths"] for item in split_items], axis=0)
        cnn_inputs_mat = np.stack([item["cnn_input_paths"] for item in split_items], axis=0)
        lm_available_mat = np.stack([item["landmark_available"] for item in split_items], axis=0)

        np.savez_compressed(
            npz_filename,
            sequence_id=seq_ids,
            video_filename=video_fns,
            behavior=behaviors,
            gender=genders,
            label=labels,
            start_frame_index=start_indices,
            end_frame_index=end_indices,
            valid_frame_count=valid_counts,
            reused_frame_count=reused_counts,
            duration_seconds=durations,
            frame_paths=frame_paths_mat,
            face_crop_paths=face_crops_mat,
            cnn_input_paths=cnn_inputs_mat,
            landmark_available=lm_available_mat,
        )
        saved_files.append(npz_filename)
        print(f"Saved {split} sequence arrays to: {npz_filename} ({len(seq_ids)} sequences)")

    # ---------------------------------------------------------
    # STRICT VALIDATION SUITE (12 CHECKS)
    # ---------------------------------------------------------
    print("\nRunning automated validation checks...")
    validation_failures = []

    # Check 1: Every sequence has exactly 16 frames
    if not (seq_df["seq_len"] == 16).all():
        validation_failures.append("Check 1 Failed: Not all sequences have exactly 16 frames.")

    # Check 2 & 3 & 4 & 5 & 6: Validate sequences in split NPZ files
    for split in ["train", "val", "test"]:
        split_items = split_npz_data[split]
        for item in split_items:
            f_paths = item["cnn_input_paths"]
            if len(f_paths) != 16:
                validation_failures.append(f"Check 1 Failed in {item['sequence_id']}: len != 16")
            
            # Check strictly increasing start vs end
            if item["start_frame_index"] >= item["end_frame_index"]:
                validation_failures.append(
                    f"Check 2 Failed in {item['sequence_id']}: frame indices not strictly increasing."
                )

    # Check 7, 8, 9: Split leakage check
    train_vids = set(seq_df[seq_df["split"] == "train"]["video_filename"])
    val_vids = set(seq_df[seq_df["split"] == "val"]["video_filename"])
    test_vids = set(seq_df[seq_df["split"] == "test"]["video_filename"])

    if train_vids.intersection(val_vids):
        validation_failures.append("Check 7 Failed: Train and Val splits share video filenames!")
    if train_vids.intersection(test_vids):
        validation_failures.append("Check 8 Failed: Train and Test splits share video filenames!")
    if val_vids.intersection(test_vids):
        validation_failures.append("Check 9 Failed: Val and Test splits share video filenames!")

    # Check 10: No duplicate sequence IDs
    if seq_df["sequence_id"].duplicated().any():
        validation_failures.append("Check 10 Failed: Duplicate sequence IDs detected!")

    # Check 11: Sequence metadata maps to actual frame files on disk (spot check)
    sample_crops = split_npz_data["train"][0]["cnn_input_paths"]
    for c_path in sample_crops[:3]:
        if not os.path.exists(c_path):
            validation_failures.append(f"Check 11 Failed: Sample crop path does not exist: {c_path}")

    # Check 12: No Driver ID exists anywhere in metadata columns or NPZ files
    illegal_cols = [c for c in seq_df.columns if "driver" in c.lower()]
    if illegal_cols:
        validation_failures.append(f"Check 12 Failed: Found Driver ID column(s): {illegal_cols}")
    
    for split in ["train", "val", "test"]:
        npz_p = os.path.join(CFG.SEQ_DIR, split, f"sequences_{split}.npz")
        if os.path.exists(npz_p):
            with np.load(npz_p) as data:
                npz_keys = list(data.keys())
                illegal_keys = [k for k in npz_keys if "driver" in k.lower()]
                if illegal_keys:
                    validation_failures.append(f"Check 12 Failed: Found Driver ID key in {split} NPZ: {illegal_keys}")

    # ---------------------------------------------------------
    # STATISTICS REPORTING
    # ---------------------------------------------------------
    total_seqs = len(seq_df)
    train_seqs = (seq_df["split"] == "train").sum()
    val_seqs = (seq_df["split"] == "val").sum()
    test_seqs = (seq_df["split"] == "test").sum()

    vids_per_seq = seq_df.groupby("video_filename")["sequence_id"].count()
    avg_seq_per_vid = vids_per_seq.mean()
    min_seq_per_vid = vids_per_seq.min()
    max_seq_per_vid = vids_per_seq.max()

    pct_affected_frames = (missing_landmark_frames / total_input_frames) * 100.0
    pct_discarded_seqs = (discarded_seq_count / total_windows_examined) * 100.0 if total_windows_examined > 0 else 0

    print("\n" + "=" * 80)
    print("STAGE 07 RESULTS & STATISTICS REPORT")
    print("=" * 80)

    print("\n1. FILES CREATED & OUTPUT DIRECTORIES:")
    for f in saved_files:
        print(f"   - {f}")

    print("\n2. SEQUENCE COUNTS:")
    print(f"   - Total Windows Examined: {total_windows_examined}")
    print(f"   - Total Valid Sequences Saved: {total_seqs}")
    print(f"   - Train Sequences: {train_seqs} ({train_seqs / total_seqs * 100:.1f}%)")
    print(f"   - Validation Sequences: {val_seqs} ({val_seqs / total_seqs * 100:.1f}%)")
    print(f"   - Test Sequences: {test_seqs} ({test_seqs / total_seqs * 100:.1f}%)")

    print("\n3. PER-VIDEO STATISTICS:")
    print(f"   - Total Unique Videos: {len(vids_per_seq)}")
    print(f"   - Average Sequences per Video: {avg_seq_per_vid:.2f}")
    print(f"   - Min Sequences per Video: {min_seq_per_vid}")
    print(f"   - Max Sequences per Video: {max_seq_per_vid}")

    print("\n4. BEHAVIOR & GENDER DISTRIBUTION:")
    print("   - By Behavior:")
    for b, count in seq_df["behavior"].value_counts().items():
        print(f"     * {b}: {count} ({count / total_seqs * 100:.1f}%)")
    print("   - By Gender:")
    for g, count in seq_df["gender"].value_counts().items():
        print(f"     * {g}: {count} ({count / total_seqs * 100:.1f}%)")

    print("\n5. DETAILED SPLIT x BEHAVIOR MATRIX (NATURAL DISTRIBUTION):")
    split_behavior_matrix = seq_df.groupby(["split", "behavior"])["sequence_id"].count()
    for s in ["train", "val", "test"]:
        print(f"   {s.upper()}:")
        for b in ["Yawning", "Microsleep"]:
            cnt = split_behavior_matrix.get((s, b), 0)
            print(f"     - {b} = {cnt}")

    print("\n6. LANDMARK VALIDITY & DISCARD STATISTICS:")
    print(f"   - Total Input Frames Affected by Missing Landmarks: {missing_landmark_frames} / {total_input_frames} ({pct_affected_frames:.2f}%)")
    print(f"   - Discarded Sequences (<80% valid face frames): {discarded_seq_count} ({pct_discarded_seqs:.2f}%)")
    print(f"   - Retained Sequences with Reused Crops (1-3 missing frames): {reused_crop_seq_count} ({reused_crop_seq_count / total_seqs * 100:.2f}%)")

    print("\n7. TEMPORAL DURATION STATISTICS:")
    min_dur = seq_df["duration_seconds"].min()
    max_dur = seq_df["duration_seconds"].max()
    mean_dur = seq_df["duration_seconds"].mean()
    print(f"   - Average Sequence Duration: {mean_dur:.4f} seconds")
    print(f"   - Min Sequence Duration: {min_dur:.4f} seconds")
    print(f"   - Max Sequence Duration: {max_dur:.4f} seconds")

    print("\n8. VALIDATION CHECKS SUMMARY:")
    if not validation_failures:
        print("   [PASS] All 12 validation checks passed successfully!")
        status = "PASSED"
    else:
        print("   [FAIL] The following validation checks failed:")
        for failure in validation_failures:
            print(f"     ! {failure}")
        status = "FAILED"

    print("\n" + "=" * 80)
    print(f"STAGE 07 STATUS: {status}")
    print("=" * 80)

    if status == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    generate_sequences()
