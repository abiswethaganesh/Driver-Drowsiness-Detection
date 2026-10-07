"""
02_dataset_inspection.py
Run this BEFORE anything else. It makes no assumptions about NITYMED's folder
layout -- it just tells you what is actually there, so you can confirm/adjust
CFG.DRIVER_ID_REGEX and the category keywords in 00_config.py.
"""

import os
import cv2
from collections import Counter

from importlib import import_module
CFG = import_module("00_config").CFG


def walk_and_report(root):
    if not os.path.isdir(root):
        print(f"!! NITYMED_ROOT does not exist: {root}")
        print("   Update CFG.NITYMED_ROOT in 00_config.py")
        return []

    video_paths = []
    print(f"Scanning {root} ...\n")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or root}/")
        for f in sorted(filenames):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(dirpath, f))
                print(f"{indent}  - {f}")
    return video_paths


def sample_video_properties(video_paths, n=5):
    print(f"\nSampling properties of up to {n} videos:\n")
    for vp in video_paths[:n]:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"  Could not open {vp}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n_frames / fps if fps else 0
        print(f"  {os.path.basename(vp)}: {w}x{h}, {fps:.1f} fps, "
              f"{n_frames} frames, ~{dur:.1f}s")
        cap.release()


def keyword_category_counts(video_paths):
    cats = Counter()
    for vp in video_paths:
        low = vp.lower()
        if any(k in low for k in CFG.YAWN_KEYWORDS):
            cats["yawning"] += 1
        elif any(k in low for k in CFG.MICROSLEEP_KEYWORDS):
            cats["microsleep"] += 1
        else:
            cats["UNRECOGNIZED"] += 1
    print("\nCategory counts inferred from path/filename keywords:")
    for k, v in cats.items():
        print(f"  {k}: {v}")
    if cats.get("UNRECOGNIZED", 0):
        print("\n!! Some files did not match yawning/microsleep keywords.")
        print("   Inspect the printed tree above and update CFG.YAWN_KEYWORDS /")
        print("   CFG.MICROSLEEP_KEYWORDS in 00_config.py accordingly.")


if __name__ == "__main__":
    CFG.ensure_dirs()
    paths = walk_and_report(CFG.NITYMED_ROOT)
    print(f"\nTotal video files found: {len(paths)}  (expected 130: 107 yawning + 21 microsleep)")
    sample_video_properties(paths)
    keyword_category_counts(paths)
    print("\nNext: open a few filenames above and confirm CFG.DRIVER_ID_REGEX "
          "in 00_config.py actually extracts a sensible driver id from them.")
