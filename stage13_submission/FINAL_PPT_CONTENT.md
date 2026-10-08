# Final Presentation Slide Content (15 Slides)

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  
**Core Technical Theme**: MobileNetV2 + LSTM Based Temporal Drowsiness Detection  
**Core Novelty**: Closed-Loop Drowsiness Intervention and Recovery Verification  

---

## SLIDE 1: TITLE & METADATA
- **Header**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation
- **Subtitle**: MobileNetV2 + LSTM Temporal Architecture with Closed-Loop Intervention & Recovery Verification
- **Course / Subject**: Neural Networks & Deep Learning (NNDL)
- **Institution**: Department of Computer Science & Engineering
- **Team**: Final Project Demonstration Group
- **Key Artifact**: Research Prototype & Reproducible Benchmark

---

## SLIDE 2: PROBLEM STATEMENT
- **The Challenge**: Industrial driver fatigue and microsleeps cause severe transportation accidents.
- **The Gap**: Traditional commercial systems focus solely on passive detection (e.g., emitting a simple chime), which drivers easily ignore or habituate to.
- **Core Insight**: Detection alone does not define what happens after an alert. Safety-critical systems require active intervention and verified driver wakefulness.
- **Our Approach**:
  $$\text{DETECT} \longrightarrow \text{ASSESS} \longrightarrow \text{INTERVENE} \longrightarrow \text{VERIFY RECOVERY} \longrightarrow \text{RESUME / ESCALATE}$$

---

## SLIDE 3: MOTIVATION
- **Industrial Transportation**: Long-haul commercial trucking and industrial mining involve extended shift durations and night-time driving.
- **Human Factors**: Cognitive fatigue impairs reaction time prior to complete eye closure.
- **Safety Objective**: Build an intelligent vision-based system that monitors temporal facial dynamics and enforces closed-loop recovery protocols before driver re-engagement.

---

## SLIDE 4: OBJECTIVES
1. **Real-Time Video Monitoring**: Capture live webcam/video frames at edge-compatible inference speeds.
2. **Temporal Dynamics Modeling**: Use a 16-frame sequence buffer to model dynamic facial changes (blinks, yawning, microsleeps) instead of isolated static frames.
3. **Deep Architecture**: Combine frozen MobileNetV2 CNN spatial feature extraction with a 64-unit LSTM temporal memory network.
4. **Closed-Loop Intervention**: Implement a severity-aware state machine with active visual/audible warnings.
5. **Recovery Verification**: Verify driver alertness ("look at camera & keep eyes open") before resuming normal monitoring.
6. **Scientific Rigor**: Systematically evaluate and report cross-domain generalization and dataset domain shift without suppressing limitations.

---

## SLIDE 5: DATASETS
- **NITYMED Dataset**:
  - Official Metadata: 130 videos across 21 drivers (11 male, 10 female; Yawning: 107, Microsleep: 21).
  - Local Usable Copy: 126 videos generating **5,822 sequences** (Train: 4,118, Val: 860, Test: 844).
  - Label Property: 100% positive drowsiness events ($\text{Label} = 1$), serving as a drowsiness recall benchmark.
- **UTA-RLDD Dataset (Local Fold-1 Subset)**:
  - 36 videos across 12 participants (3 videos per participant: Alert `0`, Low Vigilant `5`, Drowsy `10`).
  - Total Sequences: **20,009 sequences** (Train: 13,755, Val: 2,727, Test: 3,527). Diagnostic subjects: P04 & P07.
- **Mandatory Benchmark Disclaimer**:
  > *"The local UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol."*
- **Visual Asset**: `fig3_dataset_distribution.png`

---

## SLIDE 6: DATA PREPROCESSING PIPELINE
```text
Video Input / Webcam ──► Frame Stride = 3 ──► Face Detection & Landmark Extraction
                             │
                             ▼
Model Normalization (x/127.5 - 1) ◄── 16-Frame Buffer (1,16,128,128,3) ◄── 128×128 RGB Crop
```
- **Key Parameters**:
  - Image Resolution: $128 \times 128 \times 3$
  - Sequence Length: 16 frames ($530\text{ ms}$ temporal window at 30 FPS)
  - Preprocessing Normalization: $x / 127.5 - 1.0$ (MobileNetV2 Variant A1)

---

## SLIDE 7: MODEL ARCHITECTURE
- **Spatial Feature Extractor**: TimeDistributed MobileNetV2 (ImageNet weights, strictly frozen, 1280-d output per frame).
- **Temporal Sequence Modeling**: 64-Unit LSTM capturing sequential motion dynamics across 16 frames.
- **Classification Head**: Dropout(0.3) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1, Sigmoid).
- **Visual Asset**: `fig4_model_architecture.png`

---

## SLIDE 8: END-TO-END SYSTEM ARCHITECTURE
- **Complete Pipeline**:
  $$\text{Camera Input} \rightarrow \text{Face Detection} \rightarrow \text{16-Frame Buffer} \rightarrow \text{CNN-LSTM Model} \rightarrow \text{Raw Prob} \rightarrow \text{5-Window Smoothing} \rightarrow \text{State Machine} \rightarrow \text{Intervention / Recovery}$$
- **Visual Asset**: `fig1_system_architecture.png`

---

## SLIDE 9: CORE NOVELTY — CLOSED-LOOP INTERVENTION
- **Beyond Classification**:
  - Traditional Systems: $\text{Detect} \longrightarrow \text{Alert}$
  - Our Research Prototype:
    $$\text{Detect} \longrightarrow \text{Assess Persistence} \longrightarrow \text{Intervene} \longrightarrow \text{Recovery Check} \longrightarrow \text{Verify} \longrightarrow \text{Resume / Escalate}$$
- **Key Workflow Components**:
  - **WARNING State**: Sustained probability $\ge 0.5$ triggers visual/audible stop advice.
  - **RECOVERY CHECK**: Prompt: *"Look at camera & keep eyes open"* with face/eye stability verification.
  - **SAFETY REST PROTOCOL**: Simulated 3-minute rest timer countdown ($03:00$) if verification fails.
  - **Manual Emergency Alert**: Operator override button / `E` key press.
  - **Event Logging**: Automated recording to `demo_events.csv`.
- **Visual Asset**: `fig2_closed_loop_workflow.png`

---

## SLIDE 10: EXPERIMENTAL STUDY PROGRESSION
- **Stage 08 Baseline**: MobileNetV2 + LSTM frozen backbone model (`cnn_lstm_best.keras`).
- **Stage 08E Fine-Tuning**: Unfroze top 25 MobileNetV2 layers ($1\times 10^{-5}$ lr).
- **Stage 10A Domain Audit**: Identified pixel luminance/contrast shift between NITYMED & UTA-RLDD.
- **Stage 10B Domain-Aware Training**: Balanced sampling + illumination color jitter.
- **Stage 10C Domain Adaptation**: Sequence luminance normalization (A3) & temporal pooling (B2).
- **Stage 11 Evidence-Driven Selection**: Decision to **KEEP BASELINE** for reproducibility after observing participant inversion on P07 in fine-tuned models.
- **Visual Asset**: `fig7_experiment_comparison.png`

---

## SLIDE 11: QUANTITATIVE RESULTS & METRICS
- **Established Empirical Metrics**:
  - **Validation ROC-AUC**: **0.947850** | **Validation F1**: **0.965200**
  - **UTA-RLDD Test ROC-AUC**: **0.542460** | **UTA-RLDD Test F1**: **0.765152** | **UTA-RLDD FPR**: **69.02%**
  - **Combined Test ROC-AUC**: **0.638542** | **Combined Test F1**: **0.824284**
  - **NITYMED Test Recall**: **100.00%**
- **Methodological Distinction**: Validation ROC-AUC (0.947850) reflects internal split discrimination and is explicitly distinguished from external test generalization.
- **Visual Asset**: `fig5_validation_vs_external_test.png`

---

## SLIDE 12: DOMAIN SHIFT EVIDENCE & ANALYSIS
- **Illumination Gap**: NITYMED mean luminance ($69.60$) vs UTA-RLDD Alert ($116.34$).
- **Participant Inversion**: Fine-tuning caused severe inversion on participant P07 ($\text{ROC-AUC} = 0.0239$, $100\%$ FPR).
- **Key Insight**: Frozen CNN features extract background lighting characteristics, causing live webcam probability saturation ($p \approx 1.0$).
- **Visual Asset**: `fig6_domain_shift_evidence.png`

---

## SLIDE 13: REAL-TIME INFERENCE PERFORMANCE
- **Computational Benchmark**:
  - **Mean Inference Latency**: **78.65 ms** per 16-frame sequence
  - **P95 Inference Latency**: **108.86 ms**
  - **Throughput**: **15.73 FPS**
- **System Hardening**: All 17 automated system verification checks passed in `run_final_checks.py`.
- **Visual Asset**: `fig8_realtime_performance.png`

---

## SLIDE 14: LIMITATIONS & FUTURE WORK
- **Documented Limitations**:
  - Cross-domain shift & high UTA false positive rate ($69.02\%$).
  - Webcam probability saturation ($p \approx 1.0$) under bright illumination.
  - Participant-specific inversion on P07.
  - Prototype recovery check and simulated rest protocol (no vehicle control APIs).
- **Future Research Directions**:
  - Multi-source domain-adversarial training (DANN).
  - Dynamic local illumination normalization.
  - Landmark Eye Aspect Ratio (EAR) + deep probability fusion.
  - Hardware deployment on NVIDIA Jetson edge devices with IR cameras.

---

## SLIDE 15: CONCLUSION
- **Summary**: Delivered a complete, real-time driver drowsiness monitoring prototype combining temporal deep learning with a novel closed-loop intervention workflow.
- **Scientific Rigor**: Systematically investigated and reported cross-domain generalization challenges rather than hiding limitations.
- **Impact**: Provides a defensible, reproducible foundation for future safety-validated driver monitoring systems.
