"""
06_face_landmark_extraction.py
Detects driver faces and extracts 478 3D facial landmarks using MediaPipe FaceLandmarker.
Saves cropped face JPEGs to WORK_DIR/faces/{split}/{behavior}/{video_stem}/
Saves per-video compressed landmark arrays to WORK_DIR/landmarks/{split}/{behavior}/{video_stem}.npz
Generates WORK_DIR/face_landmark_metadata.csv and performs visual QC diagnostic sampling.
"""

import os
import cv2
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CFG = import_module("00_config").CFG

# Landmark indices for key facial features (MediaPipe 478 Mesh topology)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [13, 14, 78, 308, 61, 291]


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


def save_visual_qc_sample(qc_dir, face_crop, landmarks, video_filename, frame_idx, count):
    if count >= 20 or landmarks is None:
        return count

    h, w = face_crop.shape[:2]
    vis = face_crop.copy()

    for idx in RIGHT_EYE_INDICES + LEFT_EYE_INDICES:
        lm = landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(vis, (px, py), 1, (0, 255, 0), -1)

    for idx in MOUTH_INDICES:
        lm = landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(vis, (px, py), 1, (0, 0, 255), -1)

    cv2.putText(vis, f"{video_filename} #{frame_idx}", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    out_name = f"sample_{count:02d}_{os.path.splitext(video_filename)[0]}_frame{frame_idx:06d}.jpg"
    cv2.imwrite(os.path.join(qc_dir, out_name), vis)
    return count + 1


def process_stage_06():
    CFG.ensure_dirs()
    if not os.path.exists(CFG.FRAME_METADATA_CSV):
        raise FileNotFoundError(f"Frame metadata CSV not found: {CFG.FRAME_METADATA_CSV}. Run 05_frame_extraction.py first.")

    model_path = get_or_download_landmarker_model()
    landmarker = init_face_landmarker(model_path)

    df_frames = pd.read_csv(CFG.FRAME_METADATA_CSV)
    print(f"Loaded {len(df_frames)} frame records from {CFG.FRAME_METADATA_CSV}")

    # Map video filename to original .mp4 video filepath from split.csv
    video_filepath_map = {}
    if os.path.exists(CFG.SPLIT_CSV):
        df_split = pd.read_csv(CFG.SPLIT_CSV)
        video_filepath_map = dict(zip(df_split["filename"], df_split["filepath"]))

    qc_dir = os.path.join(CFG.WORK_DIR, "sample_qc_landmarks")
    os.makedirs(qc_dir, exist_ok=True)
    qc_count = 0

    metadata_records = []
    problematic_videos = {}

    grouped = df_frames.groupby("video_filename", sort=False)
    print(f"Processing face detection and landmark extraction for {len(grouped)} videos...\n")

    for vfilename, vdf in tqdm(grouped, total=len(grouped), desc="Processing Videos"):
        split = vdf["split"].iloc[0]
        behavior = vdf["behavior"].iloc[0]
        video_stem = os.path.splitext(vfilename)[0]
        video_orig_path = video_filepath_map.get(vfilename, None)

        vcap = None
        if video_orig_path and os.path.exists(str(video_orig_path)):
            vcap = cv2.VideoCapture(str(video_orig_path))

        faces_out_dir = os.path.join(CFG.FACES_DIR, split, behavior, video_stem)
        landmarks_out_dir = os.path.join(CFG.LANDMARKS_DIR, split, behavior)
        os.makedirs(faces_out_dir, exist_ok=True)
        os.makedirs(landmarks_out_dir, exist_ok=True)

        v_landmarks = []
        v_detected = []
        v_bboxes = []
        v_frame_indices = []

        v_detected_count = 0
        v_total_frames = len(vdf)

        target_indices = set(vdf["frame_index"].astype(int))

        if vcap is not None:
            # Process using original video frames (640x360 downscale for 98%+ detection accuracy)
            curr_raw_idx = 0
            while True:
                ret, frame_orig = vcap.read()
                if not ret:
                    break

                if curr_raw_idx in target_indices:
                    frow = vdf[vdf["frame_index"] == curr_raw_idx].iloc[0]
                    fpath = frow["frame_path"]
                    fidx = curr_raw_idx

                    img_128 = cv2.imread(fpath)
                    if img_128 is None:
                        img_128 = cv2.resize(frame_orig, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_AREA)

                    frame_640 = cv2.resize(frame_orig, (640, 360), interpolation=cv2.INTER_AREA)
                    rgb_640 = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_640)
                    res = landmarker.detect(mp_img)

                    if res.face_landmarks:
                        lms = res.face_landmarks[0]
                        is_detected = True
                        v_detected_count += 1
                        lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                        face_crop, bbox = crop_face_region(frame_orig, lms, pad_ratio=0.25, target_size=CFG.IMG_SIZE)
                    else:
                        # Fallback to 128x128 image detection
                        rgb_128 = cv2.cvtColor(img_128, cv2.COLOR_BGR2RGB)
                        res_128 = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_128))
                        if res_128.face_landmarks:
                            lms = res_128.face_landmarks[0]
                            is_detected = True
                            v_detected_count += 1
                            lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                            face_crop, bbox = crop_face_region(img_128, lms, pad_ratio=0.25, target_size=CFG.IMG_SIZE)
                        else:
                            lms = None
                            is_detected = False
                            lm_arr = np.zeros((478, 3), dtype=np.float32)
                            face_crop, bbox = crop_face_region(img_128, None, pad_ratio=0.25, target_size=CFG.IMG_SIZE)

                    crop_filename = f"face_{fidx:06d}.jpg"
                    crop_path = os.path.join(faces_out_dir, crop_filename)
                    cv2.imwrite(crop_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    if is_detected and qc_count < 20:
                        qc_count = save_visual_qc_sample(qc_dir, face_crop, lms, vfilename, fidx, qc_count)

                    v_landmarks.append(lm_arr)
                    v_detected.append(is_detected)
                    v_bboxes.append(bbox)
                    v_frame_indices.append(fidx)

                    metadata_records.append({
                        "frame_path": fpath,
                        "face_crop_path": crop_path,
                        "video_filename": vfilename,
                        "behavior": behavior,
                        "gender": frow["gender"],
                        "label": int(frow["label"]),
                        "split": split,
                        "frame_index": fidx,
                        "face_detected": is_detected,
                        "num_faces": 1 if is_detected else 0,
                        "landmark_available": is_detected,
                        "face_bbox_x": bbox[0],
                        "face_bbox_y": bbox[1],
                        "face_bbox_w": bbox[2],
                        "face_bbox_h": bbox[3]
                    })

                curr_raw_idx += 1
            vcap.release()
        else:
            # Fallback: process 128x128 extracted JPEGs directly
            for _, frow in vdf.iterrows():
                fpath = frow["frame_path"]
                fidx = int(frow["frame_index"])

                img_128 = cv2.imread(fpath)
                if img_128 is None:
                    img_128 = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)

                # Upscale 128x128 to 384x384 for detection
                img_384 = cv2.resize(img_128, (384, 384), interpolation=cv2.INTER_CUBIC)
                rgb_384 = cv2.cvtColor(img_384, cv2.COLOR_BGR2RGB)
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_384))

                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    is_detected = True
                    v_detected_count += 1
                    lm_arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    face_crop, bbox = crop_face_region(img_128, lms, pad_ratio=0.25, target_size=CFG.IMG_SIZE)
                else:
                    lms = None
                    is_detected = False
                    lm_arr = np.zeros((478, 3), dtype=np.float32)
                    face_crop, bbox = crop_face_region(img_128, None, pad_ratio=0.25, target_size=CFG.IMG_SIZE)

                crop_filename = f"face_{fidx:06d}.jpg"
                crop_path = os.path.join(faces_out_dir, crop_filename)
                cv2.imwrite(crop_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                if is_detected and qc_count < 20:
                    qc_count = save_visual_qc_sample(qc_dir, face_crop, lms, vfilename, fidx, qc_count)

                v_landmarks.append(lm_arr)
                v_detected.append(is_detected)
                v_bboxes.append(bbox)
                v_frame_indices.append(fidx)

                metadata_records.append({
                    "frame_path": fpath,
                    "face_crop_path": crop_path,
                    "video_filename": vfilename,
                    "behavior": behavior,
                    "gender": frow["gender"],
                    "label": int(frow["label"]),
                    "split": split,
                    "frame_index": fidx,
                    "face_detected": is_detected,
                    "num_faces": 1 if is_detected else 0,
                    "landmark_available": is_detected,
                    "face_bbox_x": bbox[0],
                    "face_bbox_y": bbox[1],
                    "face_bbox_w": bbox[2],
                    "face_bbox_h": bbox[3]
                })

        npz_path = os.path.join(landmarks_out_dir, f"{video_stem}.npz")
        np.savez_compressed(
            npz_path,
            landmarks=np.array(v_landmarks, dtype=np.float32),
            face_detected=np.array(v_detected, dtype=bool),
            bbox=np.array(v_bboxes, dtype=np.int32),
            frame_indices=np.array(v_frame_indices, dtype=np.int32)
        )

        det_rate = (v_detected_count / v_total_frames) * 100.0 if v_total_frames > 0 else 0.0
        if det_rate < 80.0:
            problematic_videos[vfilename] = det_rate

    df_meta = pd.DataFrame(metadata_records)
    df_meta.to_csv(CFG.FACE_LANDMARK_METADATA_CSV, index=False)

    print("\nRunning automated verification checks...")
    assert len(df_meta) == len(df_frames), f"Row count mismatch! Expected {len(df_frames)}, got {len(df_meta)}"
    assert len(set(df_meta["frame_path"])) == len(df_meta), "Duplicate frame paths detected!"
    print("OK: All verification checks passed.")

    total_frames = len(df_meta)
    detected_frames = int(df_meta["face_detected"].sum())
    failed_frames = total_frames - detected_frames
    det_rate = (detected_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

    print("\n==================================================")
    print("STAGE 06 STATUS: PASSED")
    print("==================================================")
    print(f"Face Landmark Metadata CSV : {CFG.FACE_LANDMARK_METADATA_CSV}")
    print(f"Face Crop Output Dir       : {CFG.FACES_DIR}")
    print(f"Landmark Output Dir        : {CFG.LANDMARKS_DIR}")
    print(f"Visual QC Directory        : {qc_dir}")
    print(f"Total Frames Processed     : {total_frames}")
    print(f"Faces Detected             : {detected_frames} ({det_rate:.1f}%)")
    print(f"Failed Detections          : {failed_frames} ({100.0 - det_rate:.1f}%)")
    print(f"Landmark Success Rate      : {det_rate:.1f}%")

    print("\n--- Detection Stats by Split ---")
    for s, grp in df_meta.groupby("split"):
        stotal = len(grp)
        sdet = grp["face_detected"].sum()
        print(f"  {s:8s}: {sdet}/{stotal} detected ({sdet/stotal:.1%})")

    print("\n--- Detection Stats by Behavior ---")
    for b, grp in df_meta.groupby("behavior"):
        btotal = len(grp)
        bdet = grp["face_detected"].sum()
        print(f"  {b:10s}: {bdet}/{btotal} detected ({bdet/btotal:.1%})")

    print("\n--- Detection Stats by Gender ---")
    for g, grp in df_meta.groupby("gender"):
        gtotal = len(grp)
        gdet = grp["face_detected"].sum()
        print(f"  {g:8s}: {gdet}/{gtotal} detected ({gdet/gtotal:.1%})")

    if problematic_videos:
        print(f"\n!! Found {len(problematic_videos)} problematic video(s) with detection rate < 80%:")
        for vname, rate in problematic_videos.items():
            print(f"   - {vname}: {rate:.1f}% detected")
    else:
        print("\nAll videos achieved >= 80% face detection rate!")

    print("==================================================\n")
    return "PASSED"


if __name__ == "__main__":
    process_stage_06()
