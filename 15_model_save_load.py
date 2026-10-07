"""
15_model_save_load.py
Small helpers for persisting the trained model and per-driver calibration
profiles (Keras 3 / TF2.x compatible .keras format + plain JSON).
"""

import os
import json
import tensorflow as tf
from importlib import import_module
CFG = import_module("00_config").CFG


def save_model(model, name="cnn_lstm_final.keras"):
    CFG.ensure_dirs()
    path = os.path.join(CFG.MODEL_DIR, name)
    model.save(path)   # .keras extension -> native Keras 3 format, TF2.x compatible
    print(f"Model saved to {path}")
    return path


def load_model(name="cnn_lstm_best.keras"):
    path = os.path.join(CFG.MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}")
    return tf.keras.models.load_model(path)


def save_calibration_profile(baseline_dict, driver_id):
    CFG.ensure_dirs()
    path = os.path.join(CFG.CALIB_DIR, f"{driver_id}.json")
    with open(path, "w") as f:
        json.dump(baseline_dict, f, indent=2)
    print(f"Calibration profile saved to {path}")
    return path


def load_calibration_profile(driver_id):
    path = os.path.join(CFG.CALIB_DIR, f"{driver_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No calibration profile at {path}")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print(f"Model dir: {CFG.MODEL_DIR}")
    print(f"Calibration dir: {CFG.CALIB_DIR}")
    if os.path.isdir(CFG.MODEL_DIR):
        print("Models found:", os.listdir(CFG.MODEL_DIR))
    if os.path.isdir(CFG.CALIB_DIR):
        print("Calibration profiles found:", os.listdir(CFG.CALIB_DIR))
