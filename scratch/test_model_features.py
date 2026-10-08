import os
import tensorflow as tf
import numpy as np

model_path = r"D:\Sem 7\NNDL\Project Demo\nitymed_work\models\cnn_lstm_best.keras"
model = tf.keras.models.load_model(model_path)
model.summary()

# Check intermediate layer for feature extraction
print("\nModel layers:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, type={type(layer).__name__}, output_shape={layer.output_shape}")

# Sub-model for 1280-d features after time_distributed_mobilenet or lstm_temporal
feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("time_distributed_mobilenet").output)
dummy = np.zeros((1, 16, 128, 128, 3), dtype=np.float32)
feat = feature_model.predict(dummy, verbose=0)
print("TimeDistributed feature shape:", feat.shape) # (1, 16, 1280)
feat_pooled = np.mean(feat, axis=1)
print("Mean pooled feature shape:", feat_pooled.shape) # (1, 1280)
