"""
05_frame_extraction.py
Extracts strided frames (every CFG.FRAME_STRIDE-th frame) from all videos in split.csv.
Resizes frames to (CFG.IMG_SIZE, CFG.IMG_SIZE), saves them as JPEGs under:
  WORK_DIR/frames/{split}/{behavior}/{video_stem}/frame_{frame_index:06d}.jpg
Generates WORK_DIR/frame_metadata.csv and performs strict integrity validation.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from importlib import import_module
CFG = import_module("00_config").CFG


def extract_frames_from_video(row, stride, img_size):
    video_path = row["filepath"]
    video_filename = row["filename"]
    split = row["split"]
    behavior = row["behavior"]
    gender = row["gender"]
    label = int(row["label"])
    source_fps = float(row["fps"])

    video_stem = os.path.splitext(video_filename)[0]
    out_dir = os.path.join(CFG.FRAMES_DIR, split, behavior, video_stem)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"!! Failed to open video: {video_path}")
        return [], 0, False

    frame_records = []
    source_frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_idx % stride == 0:
            # Resize frame to IMG_SIZE x IMG_SIZE (128x128)
            resized_frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            
            frame_filename = f"frame_{source_frame_idx:06d}.jpg"
            frame_path = os.path.join(out_dir, frame_filename)
            
            cv2.imwrite(frame_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            frame_records.append({
                "frame_path": frame_path,
                "video_filename": video_filename,
                "behavior": behavior,
                "gender": gender,
                "label": label,
                "split": split,
                "frame_index": source_frame_idx,
                "source_fps": source_fps
            })
            extracted_count += 1

        source_frame_idx += 1

    cap.release()

    if extracted_count == 0:
        print(f"!! Warning: 0 frames decoded from video: {video_path}")
        return [], 0, False

    return frame_records, extracted_count, True


def run_stage_05():
    CFG.ensure_dirs()
    if not os.path.exists(CFG.SPLIT_CSV):
        raise FileNotFoundError(f"Split CSV not found: {CFG.SPLIT_CSV}. Run 04_train_val_test_split.py first.")

    df_split = pd.read_csv(CFG.SPLIT_CSV)
    
    all_frame_records = []
    processed_count = len(df_split)
    successful_count = 0
    failed_count = 0

    video_frame_counts = []

    print(f"Extracting frames from {processed_count} videos (stride={CFG.FRAME_STRIDE}, target_size={CFG.IMG_SIZE}x{CFG.IMG_SIZE})...\n")

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc="Extracting Frames"):
        records, count, success = extract_frames_from_video(row, CFG.FRAME_STRIDE, CFG.IMG_SIZE)
        if success:
            successful_count += 1
            all_frame_records.extend(records)
            video_frame_counts.append(count)
        else:
            failed_count += 1

    # Save frame-level metadata CSV
    df_frames = pd.DataFrame(all_frame_records)
    df_frames.to_csv(CFG.FRAME_METADATA_CSV, index=False)

    # Perform automated verifications
    print("\nRunning automated integrity verification checks...")

    # 1. No split leakage in directory paths
    train_paths = set(df_frames[df_frames["split"] == "train"]["frame_path"])
    val_paths = set(df_frames[df_frames["split"] == "val"]["frame_path"])
    test_paths = set(df_frames[df_frames["split"] == "test"]["frame_path"])

    assert not (train_paths & val_paths), "Leakage detected: train frames in val set!"
    assert not (train_paths & test_paths), "Leakage detected: train frames in test set!"
    assert not (val_paths & test_paths), "Leakage detected: val frames in test set!"

    # 2. No duplicate frame paths
    assert len(df_frames) == len(set(df_frames["frame_path"])), "Duplicate frame paths detected!"

    # 3. Every successful video recorded
    recorded_videos = set(df_frames["video_filename"])
    expected_videos = set(df_split["filename"])
    assert len(recorded_videos) == successful_count, "Mismatch in recorded vs successful videos count!"

    # 4. Chronological order per video
    for vname, group in df_frames.groupby("video_filename"):
        indices = group["frame_index"].tolist()
        assert indices == sorted(indices), f"Non-chronological frame indices found for {vname}!"

    print("OK: All integrity verification checks passed successfully.")

    # Calculate statistics
    total_frames = len(df_frames)
    frames_per_split = df_frames["split"].value_counts().to_dict()
    frames_per_behavior = df_frames["behavior"].value_counts().to_dict()
    
    avg_frames = sum(video_frame_counts) / len(video_frame_counts) if video_frame_counts else 0
    min_frames = min(video_frame_counts) if video_frame_counts else 0
    max_frames = max(video_frame_counts) if video_frame_counts else 0

    status = "PASSED" if failed_count == 0 and total_frames > 0 else "FAILED"

    print("\n==================================================")
    print(f"STAGE 05 STATUS: {status}")
    print("==================================================")
    print(f"Output Directory           : {CFG.FRAMES_DIR}")
    print(f"Frame Metadata CSV         : {CFG.FRAME_METADATA_CSV}")
    print(f"Videos Processed           : {processed_count}")
    print(f"Videos Successfully Done   : {successful_count}")
    print(f"Failed Videos              : {failed_count}")
    print(f"Total Frames Extracted     : {total_frames}")
    print(f"\n--- Frames per Split ---")
    for s, c in frames_per_split.items():
        print(f"  {s:8s}: {c:6d} frames ({c/total_frames:.1%})")
    print(f"\n--- Frames per Behavior ---")
    for b, c in frames_per_behavior.items():
        print(f"  {b:10s}: {c:6d} frames ({c/total_frames:.1%})")
    print(f"\n--- Frames per Video Statistics ---")
    print(f"  Average : {avg_frames:.1f} frames")
    print(f"  Minimum : {min_frames} frames")
    print(f"  Maximum : {max_frames} frames")
    print("==================================================\n")

    return status


if __name__ == "__main__":
    run_stage_05()
