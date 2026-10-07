"""
08_cnn_lstm_model.py

CNN + LSTM architecture:
  TimeDistributed(MobileNetV2 backbone, frozen) -> per-frame 1280-d feature
  -> LSTM over the sequence -> Dense head -> sigmoid drowsiness probability.

Transfer learning: backbone starts frozen (ImageNet weights). Call
`set_backbone_trainable(model, True, n_layers=CFG.FINE_TUNE_UNFREEZE_LAYERS)`
AFTER the head has converged with the backbone frozen, then recompile with
CFG.FINE_TUNE_LEARNING_RATE and continue training a few more epochs. Doing
this from the start (fine-tuning a randomly-initialized head end-to-end)
usually destroys the pretrained features, hence the two-phase approach.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from importlib import import_module
CFG = import_module("00_config").CFG


def build_cnn_backbone(trainable=False):
    if CFG.CNN_BACKBONE == "MobileNetV2":
        base = keras.applications.MobileNetV2(
            input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3),
            include_top=False, weights="imagenet", pooling="avg",
        )
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    elif CFG.CNN_BACKBONE == "EfficientNetB0":
        base = keras.applications.EfficientNetB0(
            input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3),
            include_top=False, weights="imagenet", pooling="avg",
        )
        preprocess = keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown backbone {CFG.CNN_BACKBONE}")

    base.trainable = trainable
    return base, preprocess


def build_cnn_lstm_model():
    base, preprocess = build_cnn_backbone(trainable=False)

    seq_input = layers.Input(
        shape=(CFG.SEQ_LEN, CFG.IMG_SIZE, CFG.IMG_SIZE, 3), name="frame_sequence"
    )

    x = layers.TimeDistributed(layers.Lambda(preprocess))(seq_input)
    x = layers.TimeDistributed(base, name="td_cnn_backbone")(x)          # (B, T, 1280)
    x = layers.Masking()(x)
    x = layers.LSTM(CFG.LSTM_UNITS, return_sequences=False, name="temporal_lstm")(x)
    x = layers.Dropout(CFG.DROPOUT)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(CFG.DROPOUT / 2)(x)
    out = layers.Dense(1, activation="sigmoid", name="drowsiness_prob")(x)

    model = keras.Model(seq_input, out, name="cnn_lstm_drowsiness")
    return model


def compile_model(model, lr=None):
    lr = lr or CFG.LEARNING_RATE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def set_backbone_trainable(model, trainable, n_layers=None):
    """Optional fine-tuning step. Call after initial training with the
    backbone frozen has converged."""
    td_backbone = model.get_layer("td_cnn_backbone")
    base_cnn = td_backbone.layer  # the wrapped MobileNetV2/EfficientNetB0
    base_cnn.trainable = trainable
    if trainable and n_layers is not None:
        for layer in base_cnn.layers[:-n_layers]:
            layer.trainable = False
    return model


if __name__ == "__main__":
    m = build_cnn_lstm_model()
    m = compile_model(m)
    m.summary()
