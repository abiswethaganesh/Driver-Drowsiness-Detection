"""
09_training.py

Lazy data generator that reads sequence windows on demand from the per-video
.npz face archives (stage 06), using the sequence index built in stage 07.
An LRU-style cache keeps a handful of the most recently used videos'
face arrays in memory so we don't reopen the same .npz for every
overlapping window drawn from it.

Then: trains the CNN-LSTM model (stage 08) with the backbone frozen,
using class weights to counter the imbalance between normal and
drowsiness-indicative sequences, with EarlyStopping + ModelCheckpoint.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import OrderedDict
from importlib import import_module

CFG = import_module("00_config").CFG
model_lib = import_module("08_cnn_lstm_model")


class NpzCache:
    """Tiny LRU cache for opened .npz face arrays, keyed by path."""
    def __init__(self, max_items=8):
        self.max_items = max_items
        self._cache = OrderedDict()

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        arr = np.load(path, allow_pickle=True)["faces"]
        self._cache[path] = arr
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return arr


class SequenceGenerator(keras.utils.Sequence):
    def __init__(self, index_df, batch_size, shuffle=True, cache_size=8):
        self.df = index_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = NpzCache(cache_size)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.order = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.order)  # shuffles WHICH sequences per batch,
            # never the frames *within* a sequence -- temporal order inside
            # every window is always preserved.

    def __getitem__(self, batch_idx):
        rows = self.df.iloc[
            self.order[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        ]
        X = np.zeros((len(rows), CFG.SEQ_LEN, CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.float32)
        y = np.zeros((len(rows),), dtype=np.float32)
        for i, (_, row) in enumerate(rows.iterrows()):
            faces = self.cache.get(row["npz_path"])
            window = faces[row["start_idx"]:row["end_idx"]]
            X[i, :len(window)] = window.astype(np.float32)
            y[i] = row["label"]
        return X, y


def load_index():
    idx_csv = os.path.join(CFG.SEQ_DIR, "sequence_index.csv")
    df = pd.read_csv(idx_csv)
    return df


def compute_class_weights(train_df):
    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    total = len(train_df)
    # standard inverse-frequency weighting
    w0 = total / (2.0 * neg) if neg else 1.0
    w1 = total / (2.0 * pos) if pos else 1.0
    return {0: w0, 1: w1}


def main():
    CFG.ensure_dirs()
    df = load_index()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_gen = SequenceGenerator(train_df, CFG.BATCH_SIZE, shuffle=True)
    val_gen = SequenceGenerator(val_df, CFG.BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(train_df)
    print(f"Class weights (train): {class_weights}")

    model = model_lib.build_cnn_lstm_model()
    model = model_lib.compile_model(model)

    ckpt_path = os.path.join(CFG.MODEL_DIR, "cnn_lstm_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max",
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                           factor=0.5, patience=3, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_path = os.path.join(CFG.MODEL_DIR, "cnn_lstm_final.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path} (best checkpoint at {ckpt_path})")

    # ---- Optional fine-tuning phase (commented out by default) ----
    # Uncomment to fine-tune the last N backbone layers at a low LR once the
    # frozen-backbone model above has converged. Do this only if you have
    # Colab GPU budget left; it roughly doubles training time.
    #
    # model_lib.set_backbone_trainable(model, True, n_layers=CFG.FINE_TUNE_UNFREEZE_LAYERS)
    # model = model_lib.compile_model(model, lr=CFG.FINE_TUNE_LEARNING_RATE)
    # model.fit(train_gen, validation_data=val_gen, epochs=10,
    #           class_weight=class_weights, callbacks=callbacks)
    # model.save(os.path.join(CFG.MODEL_DIR, "cnn_lstm_finetuned.keras"))

    return history


if __name__ == "__main__":
    main()
