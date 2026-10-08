import os
import sys
import numpy as np
import cv2

WORK_DIR = r"D:\Sem 7\NNDL\Project Demo\nitymed_work"

npz_path = os.path.join(WORK_DIR, "sequences", "train", "sequences_train.npz")
data = np.load(npz_path)
face_paths = data['face_crop_paths']
print("NITYMED sample face path:", face_paths[0, 0])
p0 = face_paths[0, 0]
if not os.path.exists(p0):
    # Try joining with WORK_DIR if relative or fix Windows path
    p0_fixed = os.path.join(r"D:\Sem 7\NNDL\Project Demo", p0) if not os.path.isabs(p0) else p0
    print("Fixed path exists?", os.path.exists(p0_fixed), p0_fixed)

img = cv2.imread(p0 if os.path.exists(p0) else p0_fixed)
if img is not None:
    print("Successfully read image! Shape:", img.shape)
else:
    print("Failed to read image directly. Checking alternative paths...")

# Also check RLDD
rldd_npz_path = os.path.join(WORK_DIR, "sequences_rldd", "train", "sequences_rldd_train.npz")
rldd_data = np.load(rldd_npz_path)
rldd_paths = rldd_data['face_crop_paths']
print("UTA-RLDD sample face path:", rldd_paths[0, 0])
rp0 = rldd_paths[0, 0]
if not os.path.exists(rp0):
    rp0_fixed = os.path.join(r"D:\Sem 7\NNDL\Project Demo", rp0) if not os.path.isabs(rp0) else rp0
    print("Fixed RLDD path exists?", os.path.exists(rp0_fixed), rp0_fixed)
r_img = cv2.imread(rp0 if os.path.exists(rp0) else rp0_fixed)
if r_img is not None:
    print("Successfully read RLDD image! Shape:", r_img.shape)
