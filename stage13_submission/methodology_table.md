# Methodology & System Configuration Summary

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

| System Component | Configuration Parameter / Spec | Purpose & Justification |
| :--- | :--- | :--- |
| **Primary Datasets** | NITYMED (5,822 seqs) + UTA-RLDD (20,009 seqs) | Multi-source training and cross-domain generalization evaluation. |
| **Frame Sampling** | Stride = 3 frames | Reduces temporal redundancy while capturing motion onset. |
| **Face Localization** | MediaPipe / OpenCV Face Detection | Extracts driver facial region of interest (ROI). |
| **Input Resolution** | $128 \times 128 \times 3$ RGB | Balances visual landmark fidelity with FLOPs memory reduction. |
| **Sequence Length** | 16 frames ($530\text{ ms}$ at 30 FPS) | Models micro-blink dynamics and temporal facial behavior. |
| **CNN Backbone** | TimeDistributed MobileNetV2 (Frozen) | Lightweight spatial feature extraction ($2.25\text{M}$ params). |
| **Temporal Layer** | 64-Unit LSTM Layer | Models sequential dynamics across 16 temporal frame embeddings. |
| **Classification Head** | Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | Outputs continuous binary drowsiness probability ($p \in [0, 1]$). |
| **Loss Function** | Binary Cross-Entropy | Standard loss for single-label binary classification tasks. |
| **Optimization** | Adam ($\text{learning rate} = 1\times 10^{-4}$) | Stable stochastic gradient descent optimization. |
| **Training Split** | 17,873 sequences (NITYMED: 4,118, UTA: 13,755) | Gradient parameter update optimization pool. |
| **Validation Split** | 3,587 sequences (NITYMED: 860, UTA: 2,727) | Hyperparameter tuning and best checkpoint selection (Epoch 4). |
| **Test Split** | 4,371 sequences (NITYMED: 844, UTA: 3,527) | Single-pass held-out test evaluation (including P04 & P07). |
| **Inference Latency** | Mean: 78.65 ms \| P95: 108.86 ms | Edge-compatible CPU real-time sequence prediction. |
| **Probability Smoothing** | 5-Window Sliding Average | Filters single-frame noise and brief natural eye blinks. |
| **Intervention Machine** | 5-State Machine (`NORMAL` to `REST`) | Enforces active safety response over simple alarm chimes. |
| **Recovery Verification** | Interactive Eye Openness & Stability Check | Verifies driver wakefulness before resuming normal monitoring. |
