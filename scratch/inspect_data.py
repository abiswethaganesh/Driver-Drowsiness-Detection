import os
import sys
import pandas as pd
import numpy as np

WORK_DIR = r"D:\Sem 7\NNDL\Project Demo\nitymed_work"

print("--- Inspecting NITYMED metadata ---")
df_nity = pd.read_csv(os.path.join(WORK_DIR, "sequence_metadata.csv"))
print("NITYMED shape:", df_nity.shape)
print("NITYMED columns:", df_nity.columns.tolist())
print("NITYMED splits:", df_nity['split'].value_counts().to_dict())

print("\n--- Inspecting UTA-RLDD metadata ---")
df_rldd = pd.read_csv(os.path.join(WORK_DIR, "uta_rldd_sequence_metadata.csv"))
print("UTA-RLDD shape:", df_rldd.shape)
print("UTA-RLDD columns:", df_rldd.columns.tolist())
print("UTA-RLDD splits:", df_rldd['split'].value_counts().to_dict())
print("UTA-RLDD behaviors:", df_rldd['behavior'].value_counts().to_dict())
if 'participant_id' in df_rldd.columns:
    print("UTA-RLDD participants per split:")
    print(df_rldd.groupby('split')['participant_id'].unique().to_dict())

print("\n--- Inspecting NITYMED NPZ files ---")
for split in ['train', 'val', 'test']:
    npz_path = os.path.join(WORK_DIR, "sequences", split, f"sequences_{split}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"NITYMED {split} keys:", list(data.keys()))
        for k in data.keys():
            print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

print("\n--- Inspecting UTA-RLDD NPZ files ---")
for split in ['train', 'val', 'test']:
    npz_path = os.path.join(WORK_DIR, "sequences_rldd", split, f"sequences_rldd_{split}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"UTA-RLDD {split} keys:", list(data.keys()))
        for k in data.keys():
            print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
