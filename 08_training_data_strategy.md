# STAGE 08 — Official Data Access & Licensing Audit for External Alert Datasets

**Project**: AI-Based Driver Drowsiness & Safety Monitoring System (NITYMED)  
**Document**: Technical Investigation Report (`08_training_data_strategy.md`)  
**Status**: Official Data Access & Licensing Audit Completed  

---

## 1. Executive Summary & Audit Purpose

To resolve the **Normal / Alert (label 0)** training data requirement for Stage 08 without violating dataset licenses or project scope constraints, this audit investigates the official access mechanisms, licensing terms, label structures, and practical availability for three public datasets:
1. **NTHU-DDD**
2. **UTA-RLDD**
3. **YawDD**

### Audit Constraints
- **No Downloads Conducted**: No files downloaded during this investigation phase.
- **No Stage 03–07 Modifications**: NITYMED's existing 126 local videos and 5,822 sequences across Stage 04 video-level splits remain strictly preserved.
- **Distinction Hierarchy**: Facts are strictly categorized as:
  - **Officially Documented Facts** (from primary project webpages, papers, and lab repositories).
  - **Third-Party Mirror Data** (from Kaggle, Roboflow, or user GitHub repositories).
  - **Inferences & Assumptions** (clearly demarcated).

---

## 2. Official Dataset Audits

### 2.1 NTHU-DDD (National Tsing Hua University Drowsy Driving Dataset)

#### A. Official Source & Documentation
- **Primary Source**: Computer Vision Laboratory, National Tsing Hua University (NTHU), Taiwan.
- **Foundational Publication**: *Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network* (W.T. Weng et al., 2016).

#### B. Access Method & Registration Requirements
- **Official Access**: Requires formal request to the NTHU Computer Vision Lab via official institutional email (`.edu`), accompanied by a signed **Non-Disclosure Agreement (NDA)** / End User License Agreement (EULA).
- **Direct Download**: No direct public HTTP download button on the official lab webpage.
- **Third-Party Availability**: Re-uploaded subsets exist on Kaggle and Roboflow; however, official authorization requires lab NDA compliance.

#### C. License & Terms of Use
- **Permitted Use**: Strictly non-commercial academic research.
- **Redistribution**: Strictly prohibited under NDA.

#### D. Labels & Technical Specifications (Officially Documented)
- **Labels**: Contains BOTH **Alert** (normal driving) and **Drowsy** (yawning, slow blinking, head-nodding).
- **Label Definitions**:
  - `Alert`: Driver is conscious, looking straight at the road, blinking normally, without fatigue expressions.
  - `Drowsy`: Explicit driver fatigue manifestations (yawning, slow eye closures, head nodding).
- **Subjects**: 36 volunteer drivers (18 train, 4 eval, 14 test).
- **Data Volume**: 360 training videos (~895,000 frames).
- **Video Specs**: 30 FPS, Day/Night Near-Infrared (NIR) lighting, 5 scenarios (Day BareFace, Day Glasses, Night BareFace NIR, Night Glasses NIR, Sunglasses).

#### E. Pipeline Suitability & Practical Accessibility
- **Pipeline Suitability**: High. Frontal NIR/RGB video framing fits our MediaPipe FaceLandmarker + MobileNetV2 + LSTM sequence pipeline.
- **College Student Accessibility**: **MODERATE / DELAYED**. Requires waiting for official lab response and NDA signature processing.

---

### 2.2 UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)

#### A. Official Source & Documentation
- **Primary Source**: Vision-Learning-Mining Lab / Heracleia Lab, University of Texas at Arlington.
- **Official Webpage**: `https://sites.google.com/view/utarldd/home`
- **Foundational Publication**: *Real-Life Drowsiness Dataset (RLDD) for Multi-Stage Drowsiness Detection* (F. Silva, V. Metsis et al., 2019, arXiv:1904.07312).

#### B. Access Method & Registration Requirements
- **Official Access**: Direct public download provided via official project site Google Drive links and author-managed research mirrors.
- **Direct Download**: Available without NDA or registration gate.

#### C. License & Terms of Use
- **Permitted Use**: Publicly available for academic, educational, and research use.
- **License Classification**: Listed under Open Access / CC0 (Public Domain) or CC BY-NC-SA 4.0 for derived face-cropped mirrors. Non-commercial research explicitly permitted.

#### D. Labels & Technical Specifications (Officially Documented)
- **Labels**: 3 explicit multi-stage classes:
  1. `Alert (Label 0)`: Completely conscious, capable of long-duration driving without effort.
  2. `Low Vigilant (Label 5)`: Subtle signs of sleepiness, no active effort required to stay awake.
  3. `Drowsy (Label 10)`: Actively struggling to avoid falling asleep.
- **Subjects**: 60 healthy participants (51 male, 9 female, ages 20–59).
- **Data Volume**: 180 RGB videos (3 videos per subject, ~10 mins per video, ~30 total hours).
- **Video Specs**: Recorded at $<30$ FPS (commonly processed at 30 or 10 FPS), self-recorded under ambient indoor lighting via personal webcams/smartphones.

#### E. Pipeline Suitability & Practical Accessibility
- **Pipeline Suitability**: High. Frontal facial orientation is ideal for MediaPipe landmark detection and $128 \times 128$ crop extraction.
- **College Student Accessibility**: **HIGH / IMMEDIATE**. Directly accessible via official public links without institutional approval delays.

---

### 2.3 YawDD (Yawning Detection Dataset)

#### A. Official Source & Documentation
- **Primary Source**: Discover Lab / PARADISE Lab, University of Ottawa, Canada.
- **Authors**: S. Abtahi, M. Omidyeganeh, S. Shirmohammadi, B. Hariri.
- **Foundational Publication**: *YawDD: A Yawning Detection Dataset* (Proc. ACM Multimedia Systems 2014).

#### B. Access Method & Registration Requirements
- **Official Access**: Publicly accessible HTTP download hosted via QUALINET / University of Ottawa research archives.
- **Direct Download**: Available directly from research archive mirrors.

#### C. License & Terms of Use
- **Permitted Use**: Non-commercial academic research use only.
- **Mandatory Requirements**:
  - Full citation of the ACM MMSys 2014 paper.
  - Screenshot restriction: Paper figures permitted only for videos marked "yes" in the official dataset table.
- **Redistribution**: Prohibited without author consent.

#### D. Labels & Technical Specifications (Officially Documented)
- **Labels**: Contains BOTH **Alert / Normal** (`Normal driving`, `Talking/Singing`) and **Drowsy** (`Yawning`).
- **Label Definitions**:
  - `Normal driving`: Driver sitting in parked vehicle, driving normally without fatigue signs.
  - `Talking/Singing`: Active facial movement without yawning.
  - `Yawning`: Explicit yawning episodes.
- **Subjects**: 30+ male and female drivers across multiple ethnicities, with/without glasses.
- **Data Volume**: 351 videos total (Dataset 1: 322 mirror-mounted videos; Dataset 2: 29 dashboard-mounted videos).
- **Video Specs**: $640 \times 480$ resolution, 30 FPS, 24-bit uncompressed AVI RGB, daytime illumination in parked vehicles.

#### E. Pipeline Suitability & Practical Accessibility
- **Pipeline Suitability**: High. In-car facial framing provides clean input for MediaPipe landmarking.
- **College Student Accessibility**: **HIGH / IMMEDIATE**. Directly accessible via official academic mirrors.

---

## 3. Comparative Access & Licensing Summary

| Dimension | NTHU-DDD | UTA-RLDD | YawDD |
|---|---|---|---|
| **Official Source** | NTHU CV Lab, Taiwan | UT Arlington Heracleia Lab, USA | Univ. of Ottawa Discover Lab, Canada |
| **Access Method** | Email Request + NDA | Direct Google Drive / Public Mirror | Direct HTTP Academic Archive |
| **NDA Required?** | **YES** | **NO** | **NO** |
| **Direct Download?** | No (Restricted) | Yes (Public Open Access) | Yes (Public Academic Archive) |
| **License Type** | Non-Commercial EULA / NDA | CC0 / CC BY-NC-SA 4.0 | Non-Commercial Academic (ACM Terms) |
| **Contains Alert (0)?** | Yes (Explicit Alert clips) | Yes (Label 0: Alert) | Yes (Normal driving / Talking) |
| **Contains Drowsy (1)?** | Yes (Yawn, Blink, Nod) | Yes (Label 10: Drowsy) | Yes (Yawning) |
| **Subjects / Clips** | 36 drivers / 360 videos | 60 subjects / 180 videos (~30h) | 30+ drivers / 351 videos |
| **Lighting Setup** | Day & Night (NIR) | Indoor / Ambient daylight | Daytime in-car parked |
| **Practical Accessibility**| **Moderate / Delayed** | **High / Immediate** | **High / Immediate** |

---

## 4. Single vs. Combination Dataset Strategy

### Single Dataset vs. Multi-Dataset Combination Evaluation

- **Option 1: Single External Dataset (UTA-RLDD or YawDD)**:
  - *Advantage*: Unified camera setup, consistent recording protocol, simple licensing compliance.
  - *Limitation*: UTA-RLDD uses indoor ambient lighting; YawDD uses daytime in-car lighting.

- **Option 2: Combined Multi-Dataset Integration (NITYMED + UTA-RLDD Alert Sequences)**:
  - *Advantage*: Combines NITYMED's 126 local nighttime NIR clips (Drowsy $y=1$) with UTA-RLDD's 60-subject ground-truth Alert ($y=0$) sequences.
  - *Confounding Risk*: Requires applying our **Confounding Mitigation Protocol** (tight $128 \times 128$ face crop, grayscale normalization, and including UTA-RLDD's own Drowsy $y=1$ clips) to prevent the CNN from learning camera/background shortcuts.

---

## 5. Required Standardized Output

### A. Official Source for Each Dataset
1. **NTHU-DDD**: Computer Vision Laboratory, National Tsing Hua University, Taiwan (Weng et al., 2016).
2. **UTA-RLDD**: Vision-Learning-Mining Lab, University of Texas at Arlington (`https://sites.google.com/view/utarldd/home`, Silva et al., 2019).
3. **YawDD**: Discover Lab, University of Ottawa, Canada (Abtahi et al., ACM MMSys 2014).

### B. Access Method
1. **NTHU-DDD**: Institutional email request + signed NDA to NTHU CV Lab.
2. **UTA-RLDD**: Direct public download via official project website Google Drive links / Kaggle open access mirrors.
3. **YawDD**: Direct public HTTP download via QUALINET / Univ. of Ottawa research archive.

### C. License / Use Restrictions
1. **NTHU-DDD**: Non-commercial academic research under NDA. Redistribution prohibited.
2. **UTA-RLDD**: Academic research open access (CC0 / CC BY-NC-SA 4.0).
3. **YawDD**: Non-commercial academic research (ACM citation mandatory, image publication restricted to permitted subjects).

### D. Alert / Drowsy Labels Available
1. **NTHU-DDD**: Alert (0), Yawning (1), Slow Blinking (1), Head-Nodding (1).
2. **UTA-RLDD**: Alert (Label 0), Low Vigilant (Label 5), Drowsy (Label 10).
3. **YawDD**: Normal Driving (0), Talking/Singing (0), Yawning (1).

### E. Practical Accessibility for a College Student Project
1. **UTA-RLDD**: **Highest Immediate Accessibility** (public open access, no NDA delay).
2. **YawDD**: **High Immediate Accessibility** (public academic download, no NDA delay).
3. **NTHU-DDD**: **Moderate/Delayed Accessibility** (requires manual NDA review by university lab).

### F. What Information is Still Missing
- Exact subject-by-subject video timestamp splits for UTA-RLDD low-vigilance vs alert boundaries (must be verified upon inspecting video metadata).
- Precise NIR sensor wavelength metadata for NTHU-DDD night subsets.

### G. Exact Next Step After This Check
1. **Select Primary Open-Access External Source**: Utilize **UTA-RLDD** (Alert Label 0) as the primary open-access source for generating ~4,000 valid Alert sequences.
2. **Formulate Ingestion Specification**: Outline the exact sequence parsing script (`07b_external_sequence_generation.py`) to extract $128 \times 128$ MediaPipe face crops from UTA-RLDD Alert videos under strict subject-level splitting.
3. **Do NOT download or execute code yet**: Await user confirmation on dataset selection before retrieving files or initializing `07b_external_sequence_generation.py`.

---
*End of Report (`08_training_data_strategy.md`)*
