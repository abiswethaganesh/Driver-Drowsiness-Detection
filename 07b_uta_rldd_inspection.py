r"""
07b_uta_rldd_inspection.py
Dynamic Dataset Inspection for the UTA-RLDD (Real-Life Drowsiness Dataset) raw directory.

This script inspects the directory structure, participant folders, video files,
video containers, resolutions, framerates, total frame counts, and durations.

Target Directory: D:/Sem 7/NNDL/Project Demo/UTA_RLDD_raw/

CRITICAL: This script ONLY inspects metadata. It does NOT extract frames, run MediaPipe,
generate sequence NPZ files, or modify any files in UTA_RLDD_raw.
"""

import os
import sys
import json
import time
import cv2
from collections import defaultdict, Counter
from importlib import import_module

# Force stdout encoding to utf-8 if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Attempt to load CFG, or fallback to default paths
try:
    CFG = import_module("00_config").CFG
    CFG.ensure_dirs()
    WORK_DIR = CFG.WORK_DIR
except Exception:
    WORK_DIR = r"D:\Sem 7\NNDL\Project Demo\nitymed_work"
    os.makedirs(WORK_DIR, exist_ok=True)

UTA_RLDD_ROOT = r"D:\Sem 7\NNDL\Project Demo\UTA_RLDD_raw"


CLASS_NAME_MAP = {
    "0": "Alert (Class 0)",
    "5": "Low Vigilant (Class 5)",
    "10": "Drowsy (Class 10)"
}

LABEL_MAP = {
    "0": 0,    # Alert -> label 0 (Normal)
    "5": 1,    # Low Vigilant -> label 1 (Drowsy)
    "10": 1    # Drowsy -> label 1 (Drowsy)
}


def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def inspect_uta_rldd(root_dir):
    print("=" * 80)
    print(" UTA-RLDD (REAL-LIFE DROWSINESS DATASET) RAW DATASET INSPECTION")
    print("=" * 80)
    print(f"Target Path: {root_dir}")
    print(f"Inspection Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not os.path.exists(root_dir):
        print(f"ERROR: Target directory does not exist: {root_dir}")
        return None

    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv')
    
    video_records = []
    subject_map = defaultdict(list)
    fold_map = defaultdict(list)
    class_counter = Counter()
    corrupted_files = []
    
    total_raw_bytes = 0

    print("--- 1. Directory Tree & File Discovery ---")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        indent = "  " * depth
        
        dir_basename = os.path.basename(dirpath)
        if rel_path != ".":
            print(f"{indent}[DIR] {dir_basename}/")
            
        for f in sorted(filenames):
            full_f = os.path.join(dirpath, f)
            f_size = os.path.getsize(full_f)
            total_raw_bytes += f_size
            if f.lower().endswith(video_extensions):
                # Derive subject ID and fold if present
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
                class_key = stem if stem in CLASS_NAME_MAP else "Other"
                class_desc = CLASS_NAME_MAP.get(stem, f"Unknown ({stem})")
                binary_label = LABEL_MAP.get(stem, -1)
                
                print(f"{indent}  [VID] {f} [{format_bytes(f_size)}] (Subject: {subject_id}, State: {class_desc})")
                
                record = {
                    "full_path": full_f,
                    "rel_path": os.path.relpath(full_f, root_dir),
                    "filename": f,
                    "file_size_bytes": f_size,
                    "file_size_formatted": format_bytes(f_size),
                    "fold": fold_name,
                    "subject_id": subject_id,
                    "class_key": stem,
                    "class_name": class_desc,
                    "binary_label": binary_label
                }
                video_records.append(record)
                subject_map[subject_id].append(record)
                fold_map[fold_name].append(record)
                class_counter[class_desc] += 1

    print(f"\nDiscovered {len(video_records)} video files across {len(subject_map)} subjects.")
    print(f"Total raw directory disk size: {format_bytes(total_raw_bytes)}\n")

    print("--- 2. Video Technical Properties Extraction (OpenCV) ---")
    total_frames_all = 0
    total_duration_sec = 0.0

    for idx, rec in enumerate(video_records, 1):
        vp = rec["full_path"]
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"  [FAIL] Could not open video: {rec['rel_path']}")
            corrupted_files.append(rec["rel_path"])
            rec["is_valid"] = False
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = n_frames / fps if fps and fps > 0 else 0.0
        
        # Verify frame readability by reading first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"  [WARN] Opened but failed to read first frame: {rec['rel_path']}")
            rec["is_valid"] = False
            corrupted_files.append(rec["rel_path"])
        else:
            rec["is_valid"] = True

        rec["width"] = w
        rec["height"] = h
        rec["fps"] = round(fps, 2)
        rec["frame_count"] = n_frames
        rec["duration_sec"] = round(dur_sec, 2)

        total_frames_all += n_frames
        total_duration_sec += dur_sec

        print(f"  [{idx:02d}/{len(video_records):02d}] {rec['rel_path']:<35} | {w}x{h} | {fps:5.2f} FPS | {n_frames:6d} frames | {dur_sec:6.1f}s ({dur_sec/60:4.1f}m) | {rec['file_size_formatted']}")

    print("\n--- 3. Dataset Inspection Summary ---")
    print(f"* Total Video Count           : {len(video_records)}")
    print(f"* Valid Readable Videos       : {len(video_records) - len(corrupted_files)}")
    print(f"* Corrupted / Unreadable      : {len(corrupted_files)}")
    print(f"* Total Participants Found     : {len(subject_map)} ({', '.join(sorted(subject_map.keys()))})")
    print(f"* Folds Identified            : {len(fold_map)} ({', '.join(sorted(fold_map.keys()))})")
    print(f"* Total Aggregated Frame Count: {total_frames_all:,} frames")
    print(f"* Total Aggregated Duration   : {total_duration_sec:.1f} seconds ({total_duration_sec/60:.2f} minutes / {total_duration_sec/3600:.2f} hours)")

    print("\n--- 4. Class & State Distribution Breakdown ---")
    for cls_name, count in sorted(class_counter.items()):
        print(f"  - {cls_name:<25}: {count} videos")

    print("\n--- 5. Subject Breakdown ---")
    for subj_id in sorted(subject_map.keys()):
        s_recs = subject_map[subj_id]
        s_frames = sum(r.get("frame_count", 0) for r in s_recs if r.get("is_valid"))
        s_dur = sum(r.get("duration_sec", 0.0) for r in s_recs if r.get("is_valid"))
        s_classes = [r["class_key"] for r in s_recs]
        print(f"  * Subject {subj_id}: {len(s_recs)} videos (Classes: {sorted(s_classes)}) | {s_frames:,} frames | ~{s_dur/60:.1f} min")

    report_data = {
        "inspection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root_directory": root_dir,
        "total_video_count": len(video_records),
        "valid_video_count": len(video_records) - len(corrupted_files),
        "corrupted_files": corrupted_files,
        "total_subjects": len(subject_map),
        "subject_list": sorted(list(subject_map.keys())),
        "total_aggregated_frames": total_frames_all,
        "total_aggregated_duration_sec": total_duration_sec,
        "total_aggregated_duration_min": total_duration_sec / 60.0,
        "class_distribution": dict(class_counter),
        "video_records": video_records
    }

    report_path = os.path.join(WORK_DIR, "uta_rldd_inspection_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nInspection report JSON written to: {report_path}")
    print("=" * 80)
    print(" INSPECTION COMPLETE -- NO DATASET MODIFICATIONS WERE PERFORMED")
    print("=" * 80)
    
    return report_data


if __name__ == "__main__":
    inspect_uta_rldd(UTA_RLDD_ROOT)
