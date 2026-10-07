"""
11_baseline_system.py

The BASELINE system, as specified: CNN + LSTM + a single global decision
threshold, identical for every driver. No personalization. This is
intentionally thin -- the trained model IS the baseline system; this file
just wraps it in the same interface the personalized system (stage 12-13)
will use, so the two can be compared apples-to-apples in stage 13.
"""

import numpy as np
from importlib import import_module
CFG = import_module("00_config").CFG


class BaselineSystem:
    """CNN-LSTM raw probability compared to one fixed global threshold."""

    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold if threshold is not None else CFG.GLOBAL_THRESHOLD

    def predict_proba(self, sequence_batch):
        """sequence_batch: (B, SEQ_LEN, IMG_SIZE, IMG_SIZE, 3) uint8/float array."""
        return self.model.predict(sequence_batch, verbose=0).ravel()

    def decide(self, prob):
        return int(prob >= self.threshold)


if __name__ == "__main__":
    import tensorflow as tf
    import os
    model = tf.keras.models.load_model(os.path.join(CFG.MODEL_DIR, "cnn_lstm_best.keras"))
    baseline = BaselineSystem(model)
    dummy = np.random.rand(1, CFG.SEQ_LEN, CFG.IMG_SIZE, CFG.IMG_SIZE, 3).astype(np.float32) * 255
    p = baseline.predict_proba(dummy)[0]
    print(f"Baseline prob={p:.3f} -> decision={baseline.decide(p)}")
