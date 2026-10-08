# Stage 14 — Final Presentation & Academic Submission Package

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  
**Core Architecture**: TimeDistributed MobileNetV2 + 64-Unit LSTM  
**Core Novelty**: Closed-Loop Drowsiness Intervention and Recovery Verification  

---

## 1. Submission Package Overview

This directory (`nitymed_work/stage14_final_submission/`) contains the complete, hardened academic submission package, PowerPoint presentation (`.pptx`), research figures, report markdown files, viva defense Q&A, and automated audit logs:

- **Presentation Slide Deck**: [`Driver_Drowsiness_Final_Presentation.pptx`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/Driver_Drowsiness_Final_Presentation.pptx) (15 slides, dark navy theme, embedded figures, speaker notes on every slide).
- **Academic Project Report**: [`project_report/final_project_report.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/project_report/final_project_report.md)
- **Documentation Tables**: [`documentation/methodology_table.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/documentation/methodology_table.md), [`documentation/results_table.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/documentation/results_table.md), [`documentation/limitations.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/documentation/limitations.md), [`documentation/dataset_summary.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/documentation/dataset_summary.md)
- **Demo Assets**: [`demo/final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/demo/final_demo.py), [`demo/demo_script.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/demo/demo_script.md), [`demo/demo_checklist.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/demo/demo_checklist.md)
- **Research Figures**: [`figures/`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/figures/) (`fig1_system_architecture.png` to `fig8_realtime_performance.png`)
- **Viva Voce & Pitch**: [`viva/viva_defense.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/viva/viva_defense.md), [`viva/elevator_pitch.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/viva/elevator_pitch.md)
- **Audit Logs**: [`audits/consistency_audit.md`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/audits/consistency_audit.md), [`audits/submission_audit.json`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/audits/submission_audit.json), [`audits/ppt_quality_report.json`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage14_final_submission/audits/ppt_quality_report.json)

---

## 2. Model & Demo Artifact Locations

- **Primary Baseline Model Checkpoint**: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras)
- **Hardened Real-Time Demo Launcher**: [`nitymed_work/stage12_final_demo/final_demo.py`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_demo.py)

---

## 3. How to Run the Demonstration

### Run Final Real-Time Demo (Interactive Dashboard Mode)
```bash
python nitymed_work\stage14_final_submission\demo\final_demo.py
```

### Run Presentation Simulation Mode (For Rehearsal / Viva)
```bash
python nitymed_work\stage14_final_submission\demo\final_demo.py --demo-mode
```

### Audit PPT Quality
```bash
python nitymed_work\stage14_final_submission\check_ppt.py
```

### Audit Stage 14 Final Submission Package
```bash
python nitymed_work\stage14_final_submission\run_stage14_audit.py
```

---

## 4. Keyboard Controls during Live Demo
- `D` / `d`: Toggle Drowsiness Event Simulation
- `R` / `r`: Trigger Recovery Check State ("Look at camera & keep eyes open")
- `E` / `e`: Trigger Emergency Manual Alert
- `T` / `t`: Trigger Simulated Safety Rest Protocol Countdown
- `N` / `n`: Return System to NORMAL State
- `M` / `m`: Toggle between Live Model Mode & Presentation Demo Mode
- `Q` / `q`: Quit Demo Application & Output Logs

---

## 5. Mode Explanations
- **Live Model Mode**: Passes webcam sequence tensors directly to `cnn_lstm_best.keras` and displays raw probability output.
- **Presentation Demo Mode**: Allows presenter to manually toggle state events for clean presentation demonstration of closed-loop state machine without fake random probability hacks.

---

## 6. Safety Disclaimer
> **Research Prototype Disclaimer**: This software is an academic research prototype developed for demonstration purposes only. It is not connected to physical vehicle controls, CAN bus actuators, or braking systems, and does not claim medical diagnosis or guaranteed collision prevention.
