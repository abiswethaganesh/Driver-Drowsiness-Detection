"""
04_train_val_test_split.py
Creates a stratified video-level train / val / test split (70% / 15% / 15%).
Stratifies across combined behavior and gender categories to maintain balanced representation.
Saves the result to split.csv.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from importlib import import_module
CFG = import_module("00_config").CFG


def create_stratified_split(df, val_frac=0.15, test_frac=0.15, seed=42):
    # Combine behavior and gender into a single stratification key
    # e.g., 'Yawning_Female', 'Microsleep_Male'
    strat_key = df["behavior"].astype(str) + "_" + df["gender"].astype(str)

    # 1. First split off the test set (15%)
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        stratify=strat_key,
        random_state=seed
    )

    # Recompute strat_key for trainval subset
    trainval_strat = trainval_df["behavior"].astype(str) + "_" + trainval_df["gender"].astype(str)
    
    # 2. Split val out of trainval (15% of total dataset -> relative fraction = val_frac / (1.0 - test_frac))
    adjusted_val_frac = val_frac / (1.0 - test_frac)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=adjusted_val_frac,
        stratify=trainval_strat,
        random_state=seed
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def assert_split_integrity(full_df, train_df, val_df, test_df):
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    # Overlap assertions
    overlap_tv = train_paths & val_paths
    overlap_tt = train_paths & test_paths
    overlap_vt = val_paths & test_paths
    assert not overlap_tv, f"Leakage between train and val: {overlap_tv}"
    assert not overlap_tt, f"Leakage between train and test: {overlap_tt}"
    assert not overlap_vt, f"Leakage between val and test: {overlap_vt}"

    # Completeness assertion
    total_split_count = len(train_df) + len(val_df) + len(test_df)
    assert total_split_count == len(full_df), (
        f"Split total count ({total_split_count}) does not match metadata count ({len(full_df)})"
    )
    all_split_paths = train_paths | val_paths | test_paths
    assert all_split_paths == set(full_df["filepath"]), "Some videos from metadata are missing in split!"

    print("OK: Zero filepath overlap between train/val/test.")
    print("OK: Every metadata video appears exactly once in the split.")


def main():
    CFG.ensure_dirs()
    if not os.path.exists(CFG.METADATA_CSV):
        raise FileNotFoundError(f"Metadata file not found: {CFG.METADATA_CSV}. Run 03_video_metadata_generation.py first.")

    df = pd.read_csv(CFG.METADATA_CSV)
    
    train_df, val_df, test_df = create_stratified_split(
        df,
        val_frac=CFG.VAL_FRACTION,
        test_frac=CFG.TEST_FRACTION,
        seed=CFG.RANDOM_SEED
    )

    assert_split_integrity(df, train_df, val_df, test_df)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_split_df.to_csv(CFG.SPLIT_CSV, index=False)

    print("\n==================================================")
    print(f"Split CSV successfully created: {CFG.SPLIT_CSV}")
    print("==================================================")
    print(f"Total Video Samples : {len(full_split_df)}")
    print(f"Train Set           : {len(train_df)} ({len(train_df)/len(full_split_df):.1%})")
    print(f"Validation Set      : {len(val_df)} ({len(val_df)/len(full_split_df):.1%})")
    print(f"Test Set            : {len(test_df)} ({len(test_df)/len(full_split_df):.1%})")

    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n--- {name} Split Breakdown ({len(part)} videos) ---")
        print("Behavior Distribution:")
        print(part["behavior"].value_counts().to_string())
        print("Gender Distribution:")
        print(part["gender"].value_counts().to_string())


if __name__ == "__main__":
    main()
