"""
03_video_metadata_generation.py
Builds metadata.csv: one row per video with filepath, filename, behavior,
gender, label, resolution, fps, frame_count, duration.
"""

import os
import cv2
import pandas as pd
from importlib import import_module
CFG = import_module("00_config").CFG


def find_videos(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            if f.lower().endswith(".mp4"):
                out.append(os.path.join(dirpath, f))
    return out


def infer_metadata_from_path(filepath):
    low = filepath.lower()
    
    # Behavior derived ONLY from directory structure/keywords
    if "microsleep" in low:
        behavior = "Microsleep"
    elif "yawn" in low or "yawning" in low:
        behavior = "Yawning"
    else:
        behavior = "Unknown"
        
    # Both Microsleep and Yawning are drowsiness-indicative behaviors (label = 1)
    label = 1 if behavior in ("Microsleep", "Yawning") else 0
    
    # Gender derived from directory structure
    if "female" in low:
        gender = "Female"
    elif "male" in low:
        gender = "Male"
    else:
        gender = "Unknown"
        
    return behavior, gender, label


def probe_video(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or CFG.NATIVE_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w <= 0 or h <= 0 or n_frames <= 0:
        return None
    duration = n_frames / fps if fps else 0.0
    resolution = f"{w}x{h}"
    return fps, resolution, n_frames, duration


def build_metadata():
    CFG.ensure_dirs()
    video_paths = find_videos(CFG.NITYMED_ROOT)
    if not video_paths:
        raise RuntimeError(f"No .mp4 videos found under {CFG.NITYMED_ROOT}.")

    rows = []
    corrupted_videos = []
    
    for vp in video_paths:
        filename = os.path.basename(vp)
        behavior, gender, label = infer_metadata_from_path(vp)
        probed = probe_video(vp)
        
        if probed is None:
            print(f"!! Corrupted or unreadable video detected, skipping: {vp}")
            corrupted_videos.append(vp)
            continue
            
        fps, resolution, frame_count, duration = probed

        rows.append({
            "filepath": vp,
            "filename": filename,
            "behavior": behavior,
            "gender": gender,
            "label": label,
            "resolution": resolution,
            "fps": fps,
            "frame_count": frame_count,
            "duration": round(duration, 2)
        })

    df = pd.DataFrame(rows)
    df.to_csv(CFG.METADATA_CSV, index=False)

    print("==================================================")
    print(f"Metadata CSV successfully created: {CFG.METADATA_CSV}")
    print("==================================================")
    print(f"Total valid videos processed : {len(df)}")
    if corrupted_videos:
        print(f"Total corrupted videos skipped: {len(corrupted_videos)}")
    
    print("\n--- Behavior Distribution ---")
    print(df["behavior"].value_counts().to_string())
    
    print("\n--- Gender Distribution ---")
    print(df["gender"].value_counts().to_string())
    
    print("\n--- FPS Distribution ---")
    print(df["fps"].value_counts().to_string())
    
    print("\n--- Resolution Distribution ---")
    print(df["resolution"].value_counts().to_string())
    
    return df


if __name__ == "__main__":
    build_metadata()
