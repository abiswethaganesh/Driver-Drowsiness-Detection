# AI-Based Driver Drowsiness & Safety Monitoring System — Final Demo

## 1. Project Overview
This repository contains the final demonstration prototype for the **AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation**. The system integrates a TimeDistributed MobileNetV2 + 64-Unit LSTM deep neural network with a novel **Closed-Loop Drowsiness Intervention & Recovery Verification Workflow**.

## 2. Requirements & Dependencies
- Python 3.10+
- TensorFlow 2.11+
- OpenCV (`opencv-python`)
- NumPy
- Pandas
- Matplotlib

## 3. How to Run

### Run Final Real-Time Demo (Interactive Dashboard Mode)
```bash
python nitymed_work\stage12_final_demo\final_demo.py
```

### Run Final Demo in Simulation / Presentation Mode
```bash
python nitymed_work\stage12_final_demo\final_demo.py --demo-mode
```

### Run Automated System Final Checks (17 Verification Tests)
```bash
python nitymed_work\stage12_final_demo\run_final_checks.py
```

### Regenerate Research Figures
```bash
python nitymed_work\stage12_final_demo\generate_research_figures.py
```

## 4. Keyboard Controls during Live Demo
- `D` / `d`: Toggle Drowsiness Event Simulation
- `R` / `r`: Trigger Recovery Check State
- `E` / `e`: Trigger Manual Emergency Alert
- `T` / `t`: Trigger Simulated Safety Rest Protocol
- `N` / `n`: Return System to Normal Monitoring State
- `M` / `m`: Toggle between Live Model Mode & Presentation Demo Mode
- `Q` / `q`: Exit Application & Save Logs

## 5. System Output Files
- **Event Log**: [`nitymed_work/stage12_final_demo/demo_events.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/demo_events.csv)
- **Metrics Log**: [`nitymed_work/stage12_final_demo/realtime_metrics.csv`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/realtime_metrics.csv)
- **Check Log**: [`nitymed_work/stage12_final_demo/final_checks.json`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/final_checks.json)
- **Figures**: [`nitymed_work/stage12_final_demo/figures/`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/stage12_final_demo/figures/)

## 6. Model Path
Primary baseline model: [`nitymed_work/models/cnn_lstm_best.keras`](file:///d:/Sem%207/NNDL/Project%20Demo/nitymed_work/models/cnn_lstm_best.keras)

## 7. Safety Disclaimer
> **Research Prototype Disclaimer**: This software is an academic research prototype developed for demonstration purposes only. It is not connected to physical vehicle controls, CAN bus actuators, or braking systems, and does not claim medical diagnosis or guaranteed collision prevention.
