# Final Presentation & Demonstration Rehearsal Plan

**Project Title**: AI-Based Driver Drowsiness & Safety Monitoring System for Zero-Harm Industrial Transportation  

---

## 1. Day-Before Rehearsal Checklist

- [ ] **Run Final Demo**: Launch `python nitymed_work\stage14_final_submission\demo\final_demo.py` to confirm no import or environment errors.
- [ ] **Verify Webcam Stream**: Test webcam lighting and face cropping frame rate.
- [ ] **Verify Python Environment**: Confirm TensorFlow, OpenCV, and python-pptx are available.
- [ ] **Verify Model Checkpoint**: Confirm model file exists at `nitymed_work\models\cnn_lstm_best.keras`.
- [ ] **Test Presentation Demo Mode**: Press `M` key during demo execution to verify smooth mode switching.
- [ ] **Test Recovery Check Workflow**: Press `R` key to verify interactive recovery prompt ("Look at camera & keep eyes open").
- [ ] **Test Rest Protocol Simulation**: Press `T` key to verify simulated rest countdown ($03:00$).
- [ ] **Test Emergency Manual Alert**: Press `E` key to verify manual override logging.
- [ ] **Open PowerPoint Presentation**: Open `Driver_Drowsiness_Final_Presentation.pptx` in Microsoft PowerPoint / Impress.
- [ ] **Check Fonts & Layouts**: Verify slide typography, text alignment, and shape borders on external monitor resolution.
- [ ] **Check Image Embeddings**: Confirm figures (`fig1` through `fig8`) display crisply on slides.
- [ ] **Create Offline Backups**: Copy `.pptx` deck, demo scripts, and model checkpoint to a USB flash drive.

---

## 2. 6-Minute Complete Presentation Timeline

| Time Segment | Topic / Slide | Speaker Focus & Key Message |
| :---: | :--- | :--- |
| **0:00 – 0:30** | Slide 1–2: Title & Problem | Introduce team, fatigue hazards, and open-loop alarm limitations. |
| **0:30 – 1:00** | Slide 3: Motivation & Objectives | Explain industrial shift safety and overall system goals. |
| **1:00 – 1:40** | Slide 4–5: Datasets & Preprocessing | Describe NITYMED recall set, UTA local split, and 16-frame preprocessing. |
| **1:40 – 2:20** | Slide 6–7: Architecture | Detail MobileNetV2 + 64-unit LSTM spatio-temporal pipeline. |
| **2:20 – 3:00** | Slide 8–9: Core Novelty | Highlight CLOSED-LOOP INTERVENTION & RECOVERY VERIFICATION. |
| **3:00 – 4:00** | Slide 13 + Live Demo | Demonstrate live dashboard, state progression, and recovery check. |
| **4:00 – 4:45** | Slide 10: Results & Metrics | Present 0.9479 Val AUC, 100% NITYMED recall, and 78.65 ms latency. |
| **4:45 – 5:30** | Slide 11–12: Domain Shift & Latency | Discuss illumination shift (46.7-pt gap) and P07 diagnostic findings. |
| **5:30 – 6:00** | Slide 14–15: Limitations & Conclusion | Summarize 10 limitations, future DANN roadmap, and final conclusion. |
| **6:00+** | Q&A Defense | Answer examiner questions using `viva_defense.md` answers. |

---

## 3. Q&A Defense Strategy
- **If asked about low external ROC-AUC (0.5425)**: Emphasize illumination domain shift between dark training setups and bright test environments, pointing to Figure 6.
- **If asked about physical vehicle control**: State clearly: *"This is a research software prototype developed to demonstrate closed-loop safety workflows and is not connected to physical CAN bus actuators."*
- **If asked why baseline was retained**: Explain that fine-tuning caused severe participant inversion on subject P07 ($\text{AUC} = 0.0239$) without improving UTA ROC-AUC.
