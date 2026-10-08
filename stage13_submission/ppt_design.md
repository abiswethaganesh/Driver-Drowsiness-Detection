# Presentation Design Guidelines & Layout Recommendations

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System  

---

## 1. Visual Theme & Color Palette
- **Background**: Dark Professional Theme (`#121212` or `#1A1A1A`) for high contrast and reduced eye strain during academic slides.
- **Primary Text**: Pure White (`#FFFFFF`) / Off-White (`#E0E0E0`).
- **Accent Color 1 (Status Normal / Success)**: Emerald Green (`#2ECC71`).
- **Accent Color 2 (Status Warning / Caution)**: Safety Orange (`#F39C12`).
- **Accent Color 3 (Status Alert / Error)**: Crimson Red (`#E74C3C`).
- **Accent Color 4 (Deep Learning / Tech)**: Tech Blue (`#3498DB`).

---

## 2. Typography & Layout Rules
- **Font Family**: Modern Sans-Serif (Inter, Roboto, Arial, or Segoe UI).
- **Title Text**: 28pt – 36pt Bold.
- **Body Text**: 16pt – 20pt Regular / Medium.
- **Rule of One**: Every slide must present **one primary message**. Avoid packing multiple complex topics onto a single slide.
- **Text-to-Diagram Ratio**: Prioritize visual charts and flow diagrams over text walls.

---

## 3. Metric Callout Formatting
Key performance metrics must be presented as **large visual callouts**:
```text
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│  Validation ROC-AUC    │    │  NITYMED Test Recall   │    │ Mean Sequence Latency  │
│        0.9479          │    │        100.0%          │    │        78.65 ms        │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
```

---

## 4. Slide-by-Slide Figure Placement Map

| Slide Number | Slide Title | Recommended Figure Asset |
| :---: | :--- | :--- |
| **Slide 5** | Datasets | `figures/fig3_dataset_distribution.png` |
| **Slide 7** | Model Architecture | `figures/fig4_model_architecture.png` |
| **Slide 8** | System Architecture | `figures/fig1_system_architecture.png` |
| **Slide 9** | Novelty — Closed-Loop Workflow | `figures/fig2_closed_loop_workflow.png` |
| **Slide 10** | Experimental Study | `figures/fig7_experiment_comparison.png` |
| **Slide 11** | Results & Metrics | `figures/fig5_validation_vs_external_test.png` |
| **Slide 12** | Domain Shift Analysis | `figures/fig6_domain_shift_evidence.png` |
| **Slide 13** | Real-Time Performance | `figures/fig8_realtime_performance.png` |

---

## 5. Prohibited Practices
- **NO Fabricated Statistics**: Do not include decorative metrics (e.g., "99% accident reduction rate") that are unsupported by empirical project logs.
- **NO Cluttered Animations**: Avoid slide transitions or entry animations that distract from technical presentation.
- **NO Overwriting Metrics**: Keep all numbers identical to documented Stage 11/12 values.
