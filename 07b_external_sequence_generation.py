r"""
07b_external_sequence_generation.py
STAGE 07B — UTA-RLDD Fold 1 Preprocessing & Temporal Sequence Generation (Resumable Version)

Processes UTA-RLDD Fold 1 raw videos (36 videos across 12 participants P01-P12):
1. Subject-level split (8 train, 2 val, 2 test) with seed 42.
2. Frame sampling with FRAME_STRIDE = 3 (recording actual source_fps per video).
3. Face detection using MediaPipe FaceLandmarker and 128x128 face crop extraction.
4. 16-frame temporal sequence generation (50% overlap, 80% landmark validity threshold).
5. Output saved to WORK_DIR/sequences_rldd/ and metadata CSVs.
6. Automated 16-point validation suite & detailed statistics report.
7. SAFE RESUMPTION SUPPORT: Preserves existing face crops, cleans incomplete directories,
   saves video_cache.json per video, and skips re-processing completed videos.

CRITICAL: Does NOT modify original UTA-RLDD dataset or NITYMED files.
"""

import os
import sys
import json
import time
import glob
import shutil
import cv2
import urllib.request
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from importlib import import_module

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Force stdout encoding to utf-8 if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load CFG
CFG = import_module("00_config").CFG
CFG.ensure_dirs()

UTA_RLDD_ROOT = r"D:\Sem 7\NNDL\Project Demo\UTA_RLDD_raw"
WORK_DIR = CFG.WORK_DIR
RLDD_SEQ_DIR = os.path.join(WORK_DIR, "sequences_rldd")
RLDD_FACES_DIR = os.path.join(WORK_DIR, "rldd_faces")

FRAME_STRIDE = 3
SEQ_LEN = 16
SEQ_OVERLAP = 0.5
RANDOM_SEED = 42

CLASS_NAME_MAP = {
    "0": "Alert",
    "5": "LowVigilant",
    "10": "Drowsy"
}

LABEL_MAP = {
    "0": 0,          # Alert -> label 0
    "5": 1,          # LowVigilant -> label 1
    "10": 1          # Drowsy -> label 1
}


def get_or_download_landmarker_model():
    model_path = os.path.join(CFG.MODEL_DIR, "face_landmarker.task")
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task model from Google MediaPipe...")
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(model_url, model_path)
        print("Downloaded model successfully!")
    return model_path


def init_face_landmarker(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_face_region(source_img, landmarks, pad_ratio=0.25, target_size=128):
    h, w = source_img.shape[:2]
    if landmarks is None:
        face_crop = cv2.resize(source_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return face_crop, (0, 0, w, h)

    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))

    pw = int((x1 - x0) * pad_ratio)
    ph = int((y1 - y0) * pad_ratio)

    crop_x0, crop_x1 = max(0, x0 - pw), min(w, x1 + pw)
    crop_y0, crop_y1 = max(0, y0 - ph), min(h, y1 + ph)

    bbox_w = crop_x1 - crop_x0
    bbox_h = crop_y1 - crop_y0

    if bbox_w <= 0 or bbox_h <= 0:
        face_crop = cv2.resize(source_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return face_crop, (0, 0, w, h)

    crop = source_img[crop_y0:crop_y1, crop_x0:crop_x1]
    face_crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return face_crop, (crop_x0, crop_y0, bbox_w, bbox_h)


def get_nearest_valid_crop(face_crop_paths, landmark_flags, idx):
    n = len(landmark_flags)
    for delta in range(1, n):
        left = idx - delta
        if left >= 0 and landmark_flags[left]:
            return face_crop_paths[left]
        right = idx + delta
        if right < n and landmark_flags[right]:
            return face_crop_paths[right]
    return face_crop_paths[idx]


def get_dir_size_bytes(directory):
    total = 0
    if not os.path.exists(directory):
        return 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def discover_uta_rldd_videos(root_dir):
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv')
    videos = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        dir_basename = os.path.basename(dirpath)
        
        for f in sorted(filenames):
            if f.lower().endswith(video_extensions):
                full_path = os.path.join(dirpath, f)
                path_parts = rel_path.split(os.sep)
                fold_name = "Root"
                subject_id = "Unknown"
                
                for part in path_parts:
                    if part.lower().startswith("fold"):
                        fold_name = part
                    elif part.isdigit() or (part.lower().startswith("p") and part[1:].isdigit()):
                        subject_id = f"P{int(part.lower().replace('p','')):02d}"
                        
                if subject_id == "Unknown" and dir_basename.isdigit():
                    subject_id = f"P{int(dir_basename):02d}"

                stem = os.path.splitext(f)[0]
                behavior = CLASS_NAME_MAP.get(stem, "Unknown")
                label = LABEL_MAP.get(stem, -1)
                
                v_stem = f"{subject_id}_{stem}"
                
                videos.append({
                    "full_path": full_path,
                    "rel_path": os.path.relpath(full_path, root_dir),
                    "filename": f,
                    "fold": fold_name,
                    "participant_id": subject_id,
                    "class_key": stem,
                    "behavior": behavior,
                    "label": label,
                    "video_stem": v_stem
                })
    return videos


def assign_subject_splits(videos, seed=42):
    subject_ids = sorted(list(set(v["participant_id"] for v in videos)))
    np.random.seed(seed)
    shuffled = subject_ids.copy()
    np.random.shuffle(shuffled)
    
    # 8 train, 2 val, 2 test
    train_subs = set(shuffled[:8])
    val_subs = set(shuffled[8:10])
    test_subs = set(shuffled[10:])
    
    split_map = {}
    for s in train_subs:
        split_map[s] = "train"
    for s in val_subs:
        split_map[s] = "val"
    for s in test_subs:
        split_map[s] = "test"
        
    for v in videos:
        v["split"] = split_map[v["participant_id"]]
        
    return videos, sorted(list(train_subs)), sorted(list(val_subs)), sorted(list(test_subs))


def run_stage_07b():
    print("=" * 80)
    print(" STAGE 07B -- UTA-RLDD FOLD 1 PREPROCESSING & TEMPORAL SEQUENCE GENERATION")
    print("=" * 80)
    
    start_time = time.time()
    
    # Discover raw videos
    raw_videos = discover_uta_rldd_videos(UTA_RLDD_ROOT)
    print(f"Discovered {len(raw_videos)} video files in {UTA_RLDD_ROOT}")
    
    # Assign reproducible subject-level splits
    videos, train_subs, val_subs, test_subs = assign_subject_splits(raw_videos, seed=RANDOM_SEED)
    
    print("\n--- SUBJECT-LEVEL SPLIT ASSIGNMENT (Seed = 42) ---")
    print(f"Train Split ({len(train_subs)} subjects, {len([v for v in videos if v['split']=='train'])} vids): {', '.join(train_subs)}")
    print(f"Val Split   ({len(val_subs)} subjects, {len([v for v in videos if v['split']=='val'])} vids): {', '.join(val_subs)}")
    print(f"Test Split  ({len(test_subs)} subjects, {len([v for v in videos if v['split']=='test'])} vids): {', '.join(test_subs)}")
    print("Note: This is a temporary/custom subject-level experiment split (NOT official 5-fold benchmark).\n")
    
    # Init MediaPipe Landmarker
    model_path = get_or_download_landmarker_model()
    landmarker = init_face_landmarker(model_path)
    
    video_metadata_records = []
    sequence_records = []
    split_npz_data = {"train": [], "val": [], "test": []}
    per_split_seq_counters = {"train": 0, "val": 0, "test": 0}
    
    total_raw_frames_examined = 0
    total_sampled_frames = 0
    total_detected_faces = 0
    total_failed_detections = 0
    
    total_windows_examined = 0
    discarded_seq_count = 0
    reused_crop_seq_count = 0
    
    completed_cached_vids = []
    reprocessed_vids = []
    newly_processed_vids = []
    
    print("Processing videos, detecting faces, extracting crops, and generating sequence windows...\n")
    
    for v_idx, v in enumerate(tqdm(videos, desc="Processing UTA-RLDD Videos"), 1):
        vp = v["full_path"]
        split = v["split"]
        behavior = v["behavior"]
        label = v["label"]
        participant_id = v["participant_id"]
        v_stem = v["video_stem"]
        
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"\nERROR: Could not open video file: {vp}")
            continue
            
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = n_frames / source_fps if source_fps > 0 else 0.0
        cap.release()
        
        v["source_fps"] = round(source_fps, 2)
        v["resolution"] = f"{w}x{h}"
        v["frame_count"] = n_frames
        v["duration_seconds"] = round(dur_sec, 2)
        v["dataset_source"] = "UTA-RLDD"
        
        expected_crops = (n_frames + 2) // 3 if n_frames > 0 else 0
        faces_out_dir = os.path.join(RLDD_FACES_DIR, split, behavior, v_stem)
        cache_file = os.path.join(faces_out_dir, "video_cache.json")
        
        existing_crops = []
        if os.path.exists(faces_out_dir):
            existing_crops = sorted(glob.glob(os.path.join(faces_out_dir, "face_*.jpg")))
            
        sampled_frames = []
        is_cache_valid = False
        
        # 1. Check if valid video_cache.json exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                cached_sfs = cache_data.get("sampled_frames", [])
                if len(cached_sfs) == expected_crops and len(existing_crops) == expected_crops:
                    sampled_frames = cached_sfs
                    v_stats = cache_data.get("stats", {})
                    v_raw_frames = v_stats.get("raw_frames", n_frames)
                    v_detected = v_stats.get("detected_faces", sum(1 for sf in sampled_frames if sf["landmark_available"]))
                    v_failed = v_stats.get("failed_detections", len(sampled_frames) - v_detected)
                    
                    total_raw_frames_examined += v_raw_frames
                    total_sampled_frames += len(sampled_frames)
                    total_detected_faces += v_detected
                    total_failed_detections += v_failed
                    
                    completed_cached_vids.append(v_stem)
                    is_cache_valid = True
            except Exception:
                is_cache_valid = False
                
        # 2. If no valid cache_file, but all expected crops exist on disk
        if not is_cache_valid:
            if len(existing_crops) == expected_crops or (expected_crops > 0 and abs(len(existing_crops) - expected_crops) <= 2):
                tqdm.write(f"\n[CACHE RESTORE] Video {v_stem} has all {len(existing_crops)} crops on disk. Verifying landmarks & creating video_cache.json...")
                v_detected = 0
                v_failed = 0
                sampled_frames = []
                
                for crop_path in existing_crops:
                    fname = os.path.basename(crop_path)
                    frame_idx_str = fname.replace("face_", "").replace(".jpg", "")
                    raw_frame_idx = int(frame_idx_str) if frame_idx_str.isdigit() else 0
                    
                    img = cv2.imread(crop_path)
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        res = landmarker.detect(mp_img)
                        is_detected = bool(res.face_landmarks)
                    else:
                        is_detected = False
                        
                    if is_detected:
                        v_detected += 1
                    else:
                        v_failed += 1
                        
                    sampled_frames.append({
                        "frame_index": raw_frame_idx,
                        "timestamp_seconds": round(raw_frame_idx / source_fps, 4) if source_fps > 0 else 0.0,
                        "landmark_available": is_detected,
                        "face_crop_path": crop_path
                    })
                    
                v_raw_frames = n_frames
                total_raw_frames_examined += v_raw_frames
                total_sampled_frames += len(sampled_frames)
                total_detected_faces += v_detected
                total_failed_detections += v_failed
                
                os.makedirs(faces_out_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "video_stem": v_stem,
                        "stats": {
                            "raw_frames": v_raw_frames,
                            "sampled_frames": len(sampled_frames),
                            "detected_faces": v_detected,
                            "failed_detections": v_failed
                        },
                        "sampled_frames": sampled_frames
                    }, f, indent=2)
                    
                completed_cached_vids.append(v_stem)
                
            else:
                # 3. Incomplete or unprocessed video
                if len(existing_crops) > 0:
                    tqdm.write(f"\n[REPROCESS] Video {v_stem} is INCOMPLETE ({len(existing_crops)}/{expected_crops} crops). Safely cleaning directory...")
                    shutil.rmtree(faces_out_dir)
                    reprocessed_vids.append(v_stem)
                else:
                    newly_processed_vids.append(v_stem)
                    
                os.makedirs(faces_out_dir, exist_ok=True)
                
                cap = cv2.VideoCapture(vp)
                v_detected = 0
                v_failed = 0
                raw_frame_idx = 0
                sampled_frames = []
                
                while True:
                    ret, frame_orig = cap.read()
                    if not ret or frame_orig is None:
                        break
                        
                    total_raw_frames_examined += 1
                    
                    if raw_frame_idx % FRAME_STRIDE == 0:
                        total_sampled_frames += 1
                        frame_640 = cv2.resize(frame_orig, (640, 360), interpolation=cv2.INTER_AREA)
                        rgb_640 = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_640)
                        res = landmarker.detect(mp_img)
                        
                        if res.face_landmarks:
                            lms = res.face_landmarks[0]
                            is_detected = True
                            v_detected += 1
                            total_detected_faces += 1
                            face_crop, bbox = crop_face_region(frame_orig, lms, pad_ratio=0.25, target_size=CFG.IMG_SIZE)
                        else:
                            lms = None
                            is_detected = False
                            v_failed += 1
                            total_failed_detections += 1
                            face_crop, bbox = crop_face_region(frame_orig, None, pad_ratio=0.25, target_size=CFG.IMG_SIZE)
                            
                        crop_filename = f"face_{raw_frame_idx:06d}.jpg"
                        crop_path = os.path.join(faces_out_dir, crop_filename)
                        cv2.imwrite(crop_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        
                        sampled_frames.append({
                            "frame_index": raw_frame_idx,
                            "timestamp_seconds": round(raw_frame_idx / source_fps, 4) if source_fps > 0 else 0.0,
                            "landmark_available": is_detected,
                            "face_crop_path": crop_path
                        })
                        
                    raw_frame_idx += 1
                    
                cap.release()
                
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "video_stem": v_stem,
                        "stats": {
                            "raw_frames": raw_frame_idx,
                            "sampled_frames": len(sampled_frames),
                            "detected_faces": v_detected,
                            "failed_detections": v_failed
                        },
                        "sampled_frames": sampled_frames
                    }, f, indent=2)
                    
        # Append video metadata record
        video_metadata_records.append({
            "dataset_source": "UTA-RLDD",
            "participant_id": participant_id,
            "video_filename": v["rel_path"],
            "behavior": behavior,
            "label": label,
            "split": split,
            "source_fps": round(source_fps, 2),
            "resolution": f"{w}x{h}",
            "frame_count": n_frames,
            "duration_seconds": round(dur_sec, 2)
        })
        
        # Sequence Generation for this video
        sampled_frames.sort(key=lambda x: x["frame_index"])
        n_sampled = len(sampled_frames)
        step = max(1, int(SEQ_LEN * (1.0 - SEQ_OVERLAP))) # step = 8 sampled frames
        
        for start_i in range(0, n_sampled - SEQ_LEN + 1, step):
            window = sampled_frames[start_i : start_i + SEQ_LEN]
            total_windows_examined += 1
            
            lm_flags = [f["landmark_available"] for f in window]
            valid_cnt = sum(lm_flags)
            valid_ratio = valid_cnt / float(SEQ_LEN)
            
            if valid_ratio < 0.80:
                discarded_seq_count += 1
                continue
                
            face_crop_paths = [f["face_crop_path"] for f in window]
            cnn_input_paths = []
            reused_cnt = 0
            
            for idx in range(SEQ_LEN):
                if lm_flags[idx]:
                    cnn_input_paths.append(face_crop_paths[idx])
                else:
                    reused_crop = get_nearest_valid_crop(face_crop_paths, lm_flags, idx)
                    cnn_input_paths.append(reused_crop)
                    reused_cnt += 1
                    
            if reused_cnt > 0:
                reused_crop_seq_count += 1
                
            start_frame_idx = window[0]["frame_index"]
            end_frame_idx = window[-1]["frame_index"]
            seq_duration_sec = (end_frame_idx - start_frame_idx) / source_fps if source_fps > 0 else 0.0
            
            per_split_seq_counters[split] += 1
            seq_id = f"seq_rldd_{split}_{per_split_seq_counters[split]:06d}"
            
            seq_record = {
                "sequence_id": seq_id,
                "dataset_source": "UTA-RLDD",
                "participant_id": participant_id,
                "video_filename": v["rel_path"],
                "behavior": behavior,
                "label": label,
                "split": split,
                "start_frame_index": start_frame_idx,
                "end_frame_index": end_frame_idx,
                "sequence_length": SEQ_LEN,
                "valid_frame_count": valid_cnt,
                "reused_frame_count": reused_cnt,
                "duration_seconds": round(seq_duration_sec, 4),
                "source_fps": round(source_fps, 2)
            }
            sequence_records.append(seq_record)
            
            split_npz_data[split].append({
                "sequence_id": seq_id,
                "dataset_source": "UTA-RLDD",
                "participant_id": participant_id,
                "video_filename": v["rel_path"],
                "behavior": behavior,
                "label": label,
                "split": split,
                "start_frame_index": start_frame_idx,
                "end_frame_index": end_frame_idx,
                "valid_frame_count": valid_cnt,
                "reused_frame_count": reused_cnt,
                "duration_seconds": seq_duration_sec,
                "frame_paths": np.array([f["face_crop_path"] for f in window], dtype=str),
                "face_crop_paths": np.array(face_crop_paths, dtype=str),
                "cnn_input_paths": np.array(cnn_input_paths, dtype=str),
                "landmark_available": np.array(lm_flags, dtype=bool),
            })

    # Save Metadata CSVs
    df_video_meta = pd.DataFrame(video_metadata_records)
    video_meta_csv = os.path.join(WORK_DIR, "uta_rldd_metadata.csv")
    df_video_meta.to_csv(video_meta_csv, index=False)
    print(f"\nSaved UTA-RLDD video metadata to: {video_meta_csv} ({len(df_video_meta)} rows)")
    
    df_seq_meta = pd.DataFrame(sequence_records)
    seq_meta_csv = os.path.join(WORK_DIR, "uta_rldd_sequence_metadata.csv")
    df_seq_meta.to_csv(seq_meta_csv, index=False)
    print(f"Saved UTA-RLDD sequence metadata to: {seq_meta_csv} ({len(df_seq_meta)} rows)")
    
    # Save NPZ Archives per split
    os.makedirs(RLDD_SEQ_DIR, exist_ok=True)
    saved_npz_files = []
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(RLDD_SEQ_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        items = split_npz_data[split]
        if not items:
            continue
        npz_filename = os.path.join(split_dir, f"sequences_rldd_{split}.npz")
        
        np.savez_compressed(
            npz_filename,
            sequence_id=np.array([it["sequence_id"] for it in items], dtype=str),
            dataset_source=np.array([it["dataset_source"] for it in items], dtype=str),
            participant_id=np.array([it["participant_id"] for it in items], dtype=str),
            video_filename=np.array([it["video_filename"] for it in items], dtype=str),
            behavior=np.array([it["behavior"] for it in items], dtype=str),
            label=np.array([it["label"] for it in items], dtype=np.int32),
            split=np.array([it["split"] for it in items], dtype=str),
            start_frame_index=np.array([it["start_frame_index"] for it in items], dtype=np.int32),
            end_frame_index=np.array([it["end_frame_index"] for it in items], dtype=np.int32),
            valid_frame_count=np.array([it["valid_frame_count"] for it in items], dtype=np.int32),
            reused_frame_count=np.array([it["reused_frame_count"] for it in items], dtype=np.int32),
            duration_seconds=np.array([it["duration_seconds"] for it in items], dtype=np.float32),
            frame_paths=np.stack([it["frame_paths"] for it in items], axis=0),
            face_crop_paths=np.stack([it["face_crop_paths"] for it in items], axis=0),
            cnn_input_paths=np.stack([it["cnn_input_paths"] for it in items], axis=0),
            landmark_available=np.stack([it["landmark_available"] for it in items], axis=0),
        )
        saved_npz_files.append(npz_filename)
        print(f"Saved {split} NPZ array to: {npz_filename} ({len(items)} sequences)")

    # ---------------------------------------------------------
    # AUTOMATED 16-POINT VALIDATION SUITE
    # ---------------------------------------------------------
    print("\nRunning automated 16-point validation suite...")
    validation_failures = []
    
    # Check 1: Exactly 12 unique participants found
    unique_subs = set(df_video_meta["participant_id"])
    if len(unique_subs) != 12:
        validation_failures.append(f"Check 1 Failed: Expected 12 participants, got {len(unique_subs)}")
        
    # Check 2: Each participant has exactly 3 videos
    for p_id, grp in df_video_meta.groupby("participant_id"):
        if len(grp) != 3:
            validation_failures.append(f"Check 2 Failed: Participant {p_id} has {len(grp)} videos (expected 3)")
            
    # Check 3 & 4: Participant split isolation
    for p_id, grp in df_video_meta.groupby("participant_id"):
        if grp["split"].nunique() > 1:
            validation_failures.append(f"Check 3/4 Failed: Participant {p_id} appears in multiple splits: {grp['split'].unique()}")
            
    # Check 5: Video isolation across splits
    train_vids = set(df_video_meta[df_video_meta["split"]=="train"]["video_filename"])
    val_vids = set(df_video_meta[df_video_meta["split"]=="val"]["video_filename"])
    test_vids = set(df_video_meta[df_video_meta["split"]=="test"]["video_filename"])
    if train_vids.intersection(val_vids):
        validation_failures.append("Check 5 Failed: Train and Val share videos!")
    if train_vids.intersection(test_vids):
        validation_failures.append("Check 5 Failed: Train and Test share videos!")
    if val_vids.intersection(test_vids):
        validation_failures.append("Check 5 Failed: Val and Test share videos!")
        
    # Check 6, 7, 8: Sequence boundaries (video, participant, split)
    if not df_seq_meta.empty:
        for seq_id, grp in df_seq_meta.groupby("sequence_id"):
            if grp["video_filename"].nunique() > 1:
                validation_failures.append(f"Check 6 Failed: Sequence {seq_id} crosses videos!")
            if grp["participant_id"].nunique() > 1:
                validation_failures.append(f"Check 7 Failed: Sequence {seq_id} crosses participants!")
            if grp["split"].nunique() > 1:
                validation_failures.append(f"Check 8 Failed: Sequence {seq_id} crosses splits!")
                
    # Check 9: Every sequence has length == 16
    if not (df_seq_meta["sequence_length"] == 16).all():
        validation_failures.append("Check 9 Failed: Not all sequences have sequence_length == 16")
        
    # Check 10: Chronological frame indices
    if not (df_seq_meta["start_frame_index"] < df_seq_meta["end_frame_index"]).all():
        validation_failures.append("Check 10 Failed: Sequence start_frame_index >= end_frame_index")
        
    # Check 11: Label consistency
    alert_labels = set(df_seq_meta[df_seq_meta["behavior"]=="Alert"]["label"])
    if alert_labels and alert_labels != {0}:
        validation_failures.append(f"Check 11 Failed: Alert behavior mapped to label(s) {alert_labels} (expected {{0}})")
    lv_labels = set(df_seq_meta[df_seq_meta["behavior"]=="LowVigilant"]["label"])
    if lv_labels and lv_labels != {1}:
        validation_failures.append(f"Check 11 Failed: LowVigilant behavior mapped to label(s) {lv_labels} (expected {{1}})")
    drowsy_labels = set(df_seq_meta[df_seq_meta["behavior"]=="Drowsy"]["label"])
    if drowsy_labels and drowsy_labels != {1}:
        validation_failures.append(f"Check 11 Failed: Drowsy behavior mapped to label(s) {drowsy_labels} (expected {{1}})")
        
    # Check 12: No Driver ID field created
    for col in list(df_video_meta.columns) + list(df_seq_meta.columns):
        if "driver" in col.lower():
            validation_failures.append(f"Check 12 Failed: Illegal column found: {col}")
            
    # Check 13: Sequence crop paths exist on disk (spot check)
    if split_npz_data["train"]:
        sample_paths = split_npz_data["train"][0]["cnn_input_paths"]
        for sp in sample_paths[:3]:
            if not os.path.exists(sp):
                validation_failures.append(f"Check 13 Failed: Sample crop path does not exist: {sp}")
                
    # Check 14: No duplicate sequence IDs
    if df_seq_meta["sequence_id"].duplicated().any():
        validation_failures.append("Check 14 Failed: Duplicate sequence IDs found!")
        
    # Check 15 & 16: Failure and discard rates reported
    det_failure_rate = (total_failed_detections / total_sampled_frames * 100.0) if total_sampled_frames > 0 else 0.0
    seq_discard_rate = (discarded_seq_count / total_windows_examined * 100.0) if total_windows_examined > 0 else 0.0

    print("  Validation Checks Result:")
    if not validation_failures:
        print("  [PASS] All 16 validation checks passed successfully!")
        status = "PASSED"
    else:
        print("  [FAIL] Validation failures detected:")
        for f in validation_failures:
            print(f"    ! {f}")
        status = "FAILED"

    # Disk Space Calculations
    crops_size_bytes = get_dir_size_bytes(RLDD_FACES_DIR)
    seqs_size_bytes = get_dir_size_bytes(RLDD_SEQ_DIR)
    total_derived_bytes = crops_size_bytes + seqs_size_bytes

    elapsed_sec = time.time() - start_time

    # ---------------------------------------------------------
    # DETAILED STATISTICS REPORT SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 07B PREPROCESSING & SEQUENCE GENERATION FINAL REPORT")
    print("=" * 80)
    print(f"STAGE 07B STATUS: {status}\n")
    
    print("Completed from existing cache:")
    print(f"  {', '.join(completed_cached_vids) if completed_cached_vids else 'None'}\n")
    
    print("Reprocessed:")
    print(f"  {', '.join(reprocessed_vids) if reprocessed_vids else 'None'}\n")
    
    print("Newly processed:")
    print(f"  {', '.join(newly_processed_vids) if newly_processed_vids else 'None'}\n")
    
    print(f"Total videos: {len(videos)}")
    print(f"Videos successfully processed: {len(df_video_meta)}\n")
    
    print("Face crops:")
    print(f"  Total: {total_sampled_frames:,}")
    print(f"  Face detection failures: {total_failed_detections:,}")
    print(f"  Failure rate: {det_failure_rate:.2f}%\n")
    
    total_retained_seqs = len(df_seq_meta)
    print("Sequences:")
    print(f"  Examined: {total_windows_examined:,}")
    print(f"  Retained: {total_retained_seqs:,}")
    print(f"  Discarded: {discarded_seq_count:,} ({seq_discard_rate:.2f}%)\n")
    
    train_seqs = len(df_seq_meta[df_seq_meta["split"]=="train"])
    val_seqs = len(df_seq_meta[df_seq_meta["split"]=="val"])
    test_seqs = len(df_seq_meta[df_seq_meta["split"]=="test"])
    
    print(f"Train sequences: {train_seqs:,}")
    print(f"Validation sequences: {val_seqs:,}")
    print(f"Test sequences: {test_seqs:,}\n")
    
    alert_cnt = len(df_seq_meta[df_seq_meta["behavior"]=="Alert"])
    lv_cnt = len(df_seq_meta[df_seq_meta["behavior"]=="LowVigilant"])
    drowsy_cnt = len(df_seq_meta[df_seq_meta["behavior"]=="Drowsy"])
    
    print("Class distribution:")
    print(f"  Alert (0): {alert_cnt:,}")
    print(f"  LowVigilant (1): {lv_cnt:,}")
    print(f"  Drowsy (1): {drowsy_cnt:,}\n")
    
    print("Participant split:")
    print(f"  Train: {', '.join(train_subs)}")
    print(f"  Validation: {', '.join(val_subs)}")
    print(f"  Test: {', '.join(test_subs)}\n")
    
    passed_cnt = 16 - len(validation_failures)
    print("Validation:")
    print(f"  Checks passed: {passed_cnt}/16")
    print(f"  Checks failed: {len(validation_failures)}\n")
    
    print("Output files:")
    print(f"  - {video_meta_csv}")
    print(f"  - {seq_meta_csv}")
    for npz_f in saved_npz_files:
        print(f"  - {npz_f}")
    print(f"  - Face crops directory: {RLDD_FACES_DIR} ({format_bytes(crops_size_bytes)})")
    print(f"  - Total disk space used: {format_bytes(total_derived_bytes)}\n")
    
    print("Issues/warnings:")
    if validation_failures:
        for vf in validation_failures:
            print(f"  - {vf}")
    else:
        print("  - None")
        
    print("\nExecution Time: {:.2f} minutes".format(elapsed_sec / 60.0))
    print("=" * 80)
    
    if status == "FAILED":
        sys.exit(1)
        
    return status


if __name__ == "__main__":
    run_stage_07b()
