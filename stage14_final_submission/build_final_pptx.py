"""
STAGE 14 — POWERPOINT GENERATOR SCRIPT
Generates Driver_Drowsiness_Final_Presentation.pptx using python-pptx with dark navy theme,
embedded figures, high contrast styling, large metric callouts, and exact speaker notes.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = r"d:\Sem 7\NNDL\Project Demo"
STAGE12_FIG_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage12_final_demo\figures")
STAGE14_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage14_final_submission")
os.makedirs(STAGE14_DIR, exist_ok=True)

PPTX_PATH = os.path.join(STAGE14_DIR, "Driver_Drowsiness_Final_Presentation.pptx")

# Color Palette (Dark Navy Theme)
BG_COLOR = RGBColor(15, 23, 42)       # Dark Slate / Navy #0F172A
CARD_BG = RGBColor(30, 41, 59)        # Slate Card #1E293B
BORDER_COLOR = RGBColor(51, 65, 85)   # Border #334155
TEXT_PRIMARY = RGBColor(248, 250, 252)# White #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184)  # Muted Gray #94A3B8
ACCENT_BLUE = RGBColor(56, 189, 248)  # Cyan/Blue #38BDF8
ACCENT_GREEN = RGBColor(34, 197, 94)  # Green #22C55E
ACCENT_RED = RGBColor(239, 68, 68)    # Red #EF4444
ACCENT_ORANGE = RGBColor(245, 158, 11)# Orange #F59E0B

def set_slide_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR

def add_header(slide, title_text, category_text="DRIVER SAFETY MONITORING"):
    # Category / Tag line
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = category_text.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    # Title
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = title_text
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_PRIMARY

def add_footer(slide):
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.5), Inches(0.3))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "Research Prototype — Not a Vehicle Control System | Final Presentation"
    p.font.size = Pt(9)
    p.font.color.rgb = TEXT_MUTED

def set_speaker_notes(slide, what_to_say, key_point, potential_q):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = f"WHAT TO SAY (20-40s):\n{what_to_say}\n\nKEY POINT:\n{key_point}\n\nPOTENTIAL EXAMINER QUESTION:\n{potential_q}"

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank
    
    # =========================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)
    
    # Main Title Card
    card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.1))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = ACCENT_BLUE
    card.line.width = Pt(2)
    
    tf = card.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "NEURAL NETWORKS & DEEP LEARNING (NNDL)"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE
    p0.alignment = PP_ALIGN.CENTER
    
    p1 = tf.add_paragraph()
    p1.text = "AI-Based Driver Drowsiness & Safety Monitoring System\nfor Zero-Harm Industrial Transportation"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_PRIMARY
    p1.alignment = PP_ALIGN.CENTER
    p1.space_before = Pt(15)
    
    p2 = tf.add_paragraph()
    p2.text = "MobileNetV2 + LSTM Based Temporal Drowsiness Detection with Closed-Loop Intervention"
    p2.font.size = Pt(15)
    p2.font.color.rgb = ACCENT_GREEN
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(15)
    
    p3 = tf.add_paragraph()
    p3.text = "Team: Final Project Demonstration Group  |  Department of CSE"
    p3.font.size = Pt(12)
    p3.font.color.rgb = TEXT_MUTED
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(30)
    
    set_speaker_notes(slide1, 
                      "Good morning committee. We present our research prototype for an AI-Based Driver Drowsiness and Safety Monitoring System using a spatio-temporal MobileNetV2-LSTM architecture with closed-loop intervention.",
                      "End-to-end real-time driver drowsiness monitoring prototype.",
                      "What problem does your system solve that existing commercial systems do not?")
                      
    # =========================================================
    # SLIDE 2: PROBLEM STATEMENT
    # =========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Problem Statement: Moving Beyond Open-Loop Classification")
    add_footer(slide2)
    
    # Left Box: Traditional Approach
    b1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    b1.fill.solid()
    b1.fill.fore_color.rgb = CARD_BG
    b1.line.color.rgb = ACCENT_RED
    b1.line.width = Pt(1.5)
    tf1 = b1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "TRADITIONAL OPEN-LOOP MONITORS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED
    
    bullets1 = [
        "Focus solely on classification (Detect -> Alert)",
        "Emit simple acoustic chimes drivers easily ignore",
        "No assessment of fatigue severity or persistence",
        "Zero verification of whether driver regained alertness",
        "Open-loop gap: leaves post-alert actions undefined"
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(12)
        
    # Right Box: Our Closed-Loop Proposal
    b2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    b2.fill.solid()
    b2.fill.fore_color.rgb = CARD_BG
    b2.line.color.rgb = ACCENT_GREEN
    b2.line.width = Pt(1.5)
    tf2 = b2.text_frame
    tf2.word_wrap = True
    
    p = tf2.paragraphs[0]
    p.text = "OUR CLOSED-LOOP SAFETY WORKFLOW"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    bullets2 = [
        "Integrated pipeline: Detect -> Intervene -> Verify",
        "Evaluates temporal persistence over 16-frame buffer",
        "Active visual warning guidance instructing pull-over",
        "Mandatory Recovery Verification ('look at camera & eyes open')",
        "Escalates to 3-minute simulated rest protocol if verification fails"
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(12)
        
    set_speaker_notes(slide2,
                      "Commercial drivers habituate to simple acoustic alarms. Detection alone is insufficient for safety-critical scenarios; our system adds intervention and recovery verification.",
                      "Closed-loop workflow ensures driver wakefulness before resuming monitoring.",
                      "Why is open-loop detection insufficient?")

    # =========================================================
    # SLIDE 3: OBJECTIVES
    # =========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "System Objectives & Research Scope")
    add_footer(slide3)
    
    card3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    card3.fill.solid()
    card3.fill.fore_color.rgb = CARD_BG
    card3.line.color.rgb = BORDER_COLOR
    tf3 = card3.text_frame
    tf3.word_wrap = True
    
    objectives = [
        ("1. Real-Time Video Monitoring", "Capture live webcam/video frames at edge-compatible inference speeds."),
        ("2. Temporal Behavior Modeling", "Model facial dynamics across 16-frame temporal sequences (530 ms) rather than static isolated frames."),
        ("3. Edge Deep Architecture", "Combine frozen MobileNetV2 CNN spatial feature extraction with a 64-unit LSTM temporal memory network."),
        ("4. Closed-Loop Intervention", "Implement a severity-aware state machine issuing active visual and acoustic pull-over guidance."),
        ("5. Recovery Verification", "Verify driver eye openness and facial stability before resuming normal monitoring."),
        ("6. Event & Metric Logging", "Log all safety events to demo_events.csv and real-time latency to realtime_metrics.csv."),
        ("7. Cross-Domain Evaluation", "Rigorously evaluate cross-dataset generalization across multi-source datasets (NITYMED & UTA-RLDD).")
    ]
    
    for i, (title, desc) in enumerate(objectives):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = f"{title}: "
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        if i > 0:
            p.space_before = Pt(10)
            
        # append desc
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY
        
    set_speaker_notes(slide3,
                      "Our objectives span temporal modeling, real-time edge performance, active safety intervention, recovery verification, and transparent reporting of cross-domain limitations.",
                      "Comprehensive approach linking machine learning to real-world safety workflow.",
                      "How do you define temporal behavior?")

    # =========================================================
    # SLIDE 4: DATASETS
    # =========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Datasets & Sequence Composition")
    add_footer(slide4)
    
    # Left: NITYMED Card
    c1 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = BORDER_COLOR
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    
    p = tf_c1.paragraphs[0]
    p.text = "NITYMED DATASET (Recall Benchmark)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    nm_info = [
        "Official Metadata: 130 videos across 21 drivers (11M/10F)",
        "Official Behaviors: 107 Yawning, 21 Microsleep",
        "Local Usable Set: 126 videos -> 5,822 sequences",
        "Splits: Train (4,118), Val (860), Test (844)",
        "Label Property: 100% positive drowsiness (Label 1)"
    ]
    for info in nm_info:
        p = tf_c1.add_paragraph()
        p.text = "• " + info
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(8)
        
    # Right: UTA Card
    c2 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = BORDER_COLOR
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    
    p = tf_c2.paragraphs[0]
    p.text = "UTA-RLDD LOCAL SUBSET (Fold-1)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    uta_info = [
        "Local Subset: 36 videos across 12 participants",
        "Classes: Alert (0), Low Vigilant (5), Drowsy (10)",
        "Binary Mapping: Alert=0, Low Vigilant+Drowsy=1",
        "Total Sequences: 20,009 (Train 13,755, Val 2,727, Test 3,527)",
        "Diagnostic Held-Out Participants: P04 and P07"
    ]
    for info in uta_info:
        p = tf_c2.add_paragraph()
        p.text = "• " + info
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(8)
        
    # Footnote
    fn = slide4.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.6), Inches(0.4))
    p = fn.text_frame.paragraphs[0]
    p.text = "Methodological Disclaimer: UTA-RLDD evaluation uses a custom participant-level split of the downloaded Fold-1 subset and is not the official five-fold benchmark protocol."
    p.font.size = Pt(9)
    p.font.italic = True
    p.font.color.rgb = ACCENT_ORANGE
    
    # Embed Fig 3 if exists
    fig3_path = os.path.join(STAGE12_FIG_DIR, "fig3_dataset_distribution.png")
    if os.path.exists(fig3_path):
        slide4.shapes.add_picture(fig3_path, Inches(7.0), Inches(3.8), width=Inches(5.2))
        
    set_speaker_notes(slide4,
                      "We used NITYMED as a drowsiness recall benchmark and UTA-RLDD for multi-participant evaluation under a custom participant-level Fold-1 split.",
                      "Clear methodological distinction between NITYMED recall data and UTA-RLDD evaluation.",
                      "Is your UTA-RLDD evaluation the official five-fold benchmark?")

    # =========================================================
    # SLIDE 5: PREPROCESSING PIPELINE
    # =========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Data Preprocessing & Sequence Generation")
    add_footer(slide5)
    
    steps = [
        ("1. Video Stream", "Raw MP4/webcam feed"),
        ("2. Stride Sampling", "Frame stride = 3"),
        ("3. Face Detection", "Localize face region"),
        ("4. Face Crop", "128×128 RGB resolution"),
        ("5. Temporal Buffer", "16 frames per sequence"),
        ("6. Normalization", "x / 127.5 - 1.0 (A1)")
    ]
    
    for i, (title, desc) in enumerate(steps):
        x = Inches(0.8 + (i % 3) * 3.9)
        y = Inches(1.8 + (i // 3) * 2.4)
        
        box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.7), Inches(2.0))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = ACCENT_BLUE
        
        tf_b = box.text_frame
        tf_b.word_wrap = True
        p = tf_b.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        
        p2 = tf_b.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_PRIMARY
        p2.space_before = Pt(8)
        
    set_speaker_notes(slide5,
                      "Video is sampled at stride 3, localized to face crops, resized to 128x128, and normalized to [-1, 1] across 16 temporal frames.",
                      "Lightweight preprocessing pipeline optimized for edge inference.",
                      "Why 16 frames at stride 3?")

    # =========================================================
    # SLIDE 6: MODEL ARCHITECTURE
    # =========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Deep Neural Network Architecture (MobileNetV2 + LSTM)")
    add_footer(slide6)
    
    # Left Text Box
    t_box = slide6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(4.8))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    
    arch_details = [
        ("TimeDistributed MobileNetV2", "Pretrained ImageNet CNN, strictly frozen, outputs 1280-d spatial activations per frame."),
        ("Global Average Pooling", "Reduces 2D feature maps into compact 1D frame embeddings."),
        ("64-Unit LSTM Layer", "Models sequential temporal dependencies and facial motion dynamics across 16 frames."),
        ("Dropout Regularization", "Dropout(0.3) after LSTM and Dropout(0.2) after Dense layer to prevent overfitting."),
        ("Dense Output Layer", "Dense(32, ReLU) followed by Dense(1, Sigmoid) outputting drowsiness probability p in [0, 1].")
    ]
    for i, (head, body) in enumerate(arch_details):
        p = tf_t.paragraphs[0] if i == 0 else tf_t.add_paragraph()
        p.text = f"• {head}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_GREEN
        if i > 0:
            p.space_before = Pt(10)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY

    # Right Image
    fig4_path = os.path.join(STAGE12_FIG_DIR, "fig4_model_architecture.png")
    if os.path.exists(fig4_path):
        slide6.shapes.add_picture(fig4_path, Inches(6.5), Inches(2.0), width=Inches(6.0))
        
    set_speaker_notes(slide6,
                      "Spatial features are extracted frame-by-frame by a frozen MobileNetV2 backbone and modeled sequentially by a 64-unit LSTM network.",
                      "Hybrid spatio-temporal deep neural network.",
                      "Why frozen MobileNetV2 instead of fine-tuning?")

    # =========================================================
    # SLIDE 7: SYSTEM ARCHITECTURE
    # =========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "End-to-End System Architecture")
    add_footer(slide7)
    
    fig1_path = os.path.join(STAGE12_FIG_DIR, "fig1_system_architecture.png")
    if os.path.exists(fig1_path):
        slide7.shapes.add_picture(fig1_path, Inches(0.8), Inches(1.8), width=Inches(11.7))
        
    # Text below diagram
    t_box7 = slide7.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.5))
    tf7 = t_box7.text_frame
    tf7.word_wrap = True
    p = tf7.paragraphs[0]
    p.text = "Pipeline Flow Summary:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    p2 = tf7.add_paragraph()
    p2.text = "Camera Input -> Face Detection -> 16-Frame Sequence Buffer -> CNN-LSTM Model -> Raw Probability -> 5-Window Temporal Smoothing -> Closed-Loop State Machine -> Intervention & Recovery Verification"
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_PRIMARY
    p2.space_before = Pt(5)
    
    set_speaker_notes(slide7,
                      "The end-to-end system receives camera input, maintains a 16-frame rolling buffer, computes smoothed probabilities, and manages state transitions.",
                      "Real-time modular architecture.",
                      "Why apply temporal probability smoothing?")

    # =========================================================
    # SLIDE 8: CORE NOVELTY — CLOSED-LOOP INTERVENTION
    # =========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Core Novelty: Closed-Loop Intervention & Recovery Verification")
    add_footer(slide8)
    
    fig2_path = os.path.join(STAGE12_FIG_DIR, "fig2_closed_loop_workflow.png")
    if os.path.exists(fig2_path):
        slide8.shapes.add_picture(fig2_path, Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    t_box8 = slide8.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    tf8 = t_box8.text_frame
    tf8.word_wrap = True
    
    p = tf8.paragraphs[0]
    p.text = "WORKFLOW STAGES"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    novelty_points = [
        ("WARNING State", "Visual/audible pull-over alert issued after 3 consecutive high-probability predictions."),
        ("RECOVERY CHECK", "Prompts: 'Look at camera & keep eyes open'. Evaluates face presence and eye stability over 3s."),
        ("RECOVERY VERIFIED", "Returns system to NORMAL monitoring upon verified wakefulness."),
        ("SAFETY REST PROTOCOL", "Simulated 3-minute rest countdown (03:00) if verification fails."),
        ("Manual Emergency Alert", "Operator override button / 'E' key press logged to demo_events.csv.")
    ]
    for i, (title, desc) in enumerate(novelty_points):
        p = tf8.add_paragraph()
        p.text = f"• {title}: "
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY
        
    set_speaker_notes(slide8,
                      "Our novelty transitions the system from passive classification to active intervention and verification, requiring driver wakefulness confirmation before resuming.",
                      "Closed-loop workflow is the core project contribution.",
                      "What happens if recovery verification fails?")

    # =========================================================
    # SLIDE 9: EXPERIMENTAL STUDY PROGRESSION
    # =========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Experimental Progression Across Development Stages")
    add_footer(slide9)
    
    fig7_path = os.path.join(STAGE12_FIG_DIR, "fig7_experiment_comparison.png")
    if os.path.exists(fig7_path):
        slide9.shapes.add_picture(fig7_path, Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    t_box9 = slide9.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    tf9 = t_box9.text_frame
    tf9.word_wrap = True
    
    p = tf9.paragraphs[0]
    p.text = "EXPERIMENTAL HIGHLIGHTS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    exp_points = [
        ("Stage 08 Baseline", "MobileNetV2 + LSTM frozen model (Val AUC: 0.9479)."),
        ("Stage 08E Fine-Tuning", "Unfroze top 25 layers; observed severe P07 participant inversion (AUC 0.0339)."),
        ("Stage 10A Domain Audit", "Identified 46.7-point luminance gap between NITYMED & UTA-RLDD."),
        ("Stage 10B Domain Training", "Color jitter reduced UTA FPR to 60.30% but did not resolve CNN feature shift."),
        ("Stage 10C Domain Adaptation", "Sequence norm (A3) reduced FPR to 25.40% but degraded Val AUC to 0.6850."),
        ("Stage 11 Model Selection", "Decision: KEEP BASELINE to preserve uncorrupted reproducibility.")
    ]
    for title, desc in exp_points:
        p = tf9.add_paragraph()
        p.text = f"• {title}: "
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = ACCENT_GREEN
        p.space_before = Pt(6)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY
        
    set_speaker_notes(slide9,
                      "We conducted systematic experiments unfreezing layers and applying color jitter, retaining the baseline to preserve uncorrupted reproducibility.",
                      "Evidence-driven model evaluation.",
                      "Why did you keep the baseline instead of a fine-tuned model?")

    # =========================================================
    # SLIDE 10: QUANTITATIVE RESULTS & METRICS
    # =========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Empirical Baseline Results & Metrics")
    add_footer(slide10)
    
    metrics = [
        ("0.947850", "Validation ROC-AUC", ACCENT_GREEN),
        ("0.542460", "UTA Test ROC-AUC", ACCENT_ORANGE),
        ("0.765152", "UTA Test F1-Score", ACCENT_BLUE),
        ("69.02%", "UTA Test FPR (t=0.5)", ACCENT_RED),
        ("0.638542", "Combined Test ROC-AUC", ACCENT_BLUE),
        ("100.00%", "NITYMED Test Recall", ACCENT_GREEN)
    ]
    
    for i, (val, label, col) in enumerate(metrics):
        x = Inches(0.8 + (i % 3) * 3.9)
        y = Inches(1.8 + (i // 3) * 2.3)
        
        card = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.7), Inches(2.0))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = col
        card.line.width = Pt(1.5)
        
        tf_c = card.text_frame
        tf_c.word_wrap = True
        
        p = tf_c.paragraphs[0]
        p.text = val
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        
        p2 = tf_c.add_paragraph()
        p2.text = label
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_PRIMARY
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(8)

    set_speaker_notes(slide10,
                      "Our model achieved 0.9479 validation ROC-AUC and 100% NITYMED recall. External UTA testing yielded 0.5425 ROC-AUC, reflecting domain distribution differences.",
                      "Transparent distinction between validation and external test metrics.",
                      "Why is external UTA ROC-AUC lower than validation?")

    # =========================================================
    # SLIDE 11: DOMAIN SHIFT EVIDENCE
    # =========================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "Domain Shift Investigation & Findings")
    add_footer(slide11)
    
    fig6_path = os.path.join(STAGE12_FIG_DIR, "fig6_domain_shift_evidence.png")
    if os.path.exists(fig6_path):
        slide11.shapes.add_picture(fig6_path, Inches(0.8), Inches(1.8), width=Inches(6.2))
        
    t_box11 = slide11.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8))
    tf11 = t_box11.text_frame
    tf11.word_wrap = True
    
    p = tf11.paragraphs[0]
    p.text = "KEY DOMAIN SHIFT FINDINGS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED
    
    shift_points = [
        ("Illumination Gap", "NITYMED mean luminance is 69.60, whereas UTA-RLDD Alert mean luminance is 116.34 (46.7 point shift)."),
        ("CNN Brightness Sensitivity", "Frozen ImageNet CNN features extract overall background brightness, causing elevated probabilities on bright sequences."),
        ("Participant Sensitivity", "Participants P04 and P07 exhibited contrasting behavior, with P07 suffering complete feature inversion in fine-tuned models."),
        ("Scientific Conclusion", "The experiments indicate substantial cross-domain generalization challenges that require multi-modal feature fusion.")
    ]
    for title, desc in shift_points:
        p = tf11.add_paragraph()
        p.text = f"• {title}: "
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = ACCENT_ORANGE
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY
        
    set_speaker_notes(slide11,
                      "Domain audit revealed a 46.7-point luminance shift between datasets, causing frozen CNN features to elevate under bright lighting.",
                      "Illumination and camera setup drive cross-domain variation.",
                      "Did you solve domain shift?")

    # =========================================================
    # SLIDE 12: REAL-TIME PERFORMANCE BENCHMARK
    # =========================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "Real-Time Inference Performance & Final Checks")
    add_footer(slide12)
    
    # Left Callout Cards
    c_lat = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.2))
    c_lat.fill.solid()
    c_lat.fill.fore_color.rgb = CARD_BG
    c_lat.line.color.rgb = ACCENT_BLUE
    tf_l = c_lat.text_frame
    
    p = tf_l.paragraphs[0]
    p.text = "78.65 ms"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.alignment = PP_ALIGN.CENTER
    p2 = tf_l.add_paragraph()
    p2.text = "Mean Sequence Latency (P95: 108.86 ms | Throughput: 15.73 FPS)"
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_PRIMARY
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(8)
    
    # Right Image
    fig8_path = os.path.join(STAGE12_FIG_DIR, "fig8_realtime_performance.png")
    if os.path.exists(fig8_path):
        slide12.shapes.add_picture(fig8_path, Inches(6.8), Inches(1.8), width=Inches(5.7))
        
    # Bottom Status Check Card
    c_chk = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.3), Inches(5.6), Inches(2.3))
    c_chk.fill.solid()
    c_chk.fill.fore_color.rgb = CARD_BG
    c_chk.line.color.rgb = ACCENT_GREEN
    tf_chk = c_chk.text_frame
    tf_chk.word_wrap = True
    
    p = tf_chk.paragraphs[0]
    p.text = "AUTOMATED SYSTEM CHECKS (17/17 PASSED)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    chks = [
        "Webcam Stream: PASS", "Face Detection: PASS",
        "Temporal Inference: PASS", "State Machine: PASS",
        "Event Logging: PASS", "Performance Metrics: PASS"
    ]
    for chk in chks:
        p = tf_chk.add_paragraph()
        p.text = "✓ " + chk
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(3)
        
    set_speaker_notes(slide12,
                      "Real-time benchmarking on native CPU achieved 78.65 ms sequence latency and 15.73 FPS, passing all 17 automated system checks.",
                      "Edge-compatible CPU execution speed.",
                      "Can 78.65 ms sequence latency support real-time execution?")

    # =========================================================
    # SLIDE 13: LIVE CLOSED-LOOP DEMO
    # =========================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13)
    add_header(slide13, "Final Real-Time Demonstration & UI Dashboard")
    add_footer(slide13)
    
    c13 = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8))
    c13.fill.solid()
    c13.fill.fore_color.rgb = CARD_BG
    c13.line.color.rgb = ACCENT_BLUE
    tf13 = c13.text_frame
    tf13.word_wrap = True
    
    p = tf13.paragraphs[0]
    p.text = "DEMONSTRATION FEATURES & DASHBOARD PANELS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    
    feats = [
        ("Dual-Panel Interface", "Left Panel: Live video feed with face bounding box. Right Panel: Real-time status dashboard."),
        ("Dashboard Fields", "Displays System State, Smoothed Probability %, Face Detection Status, Temporal Buffer (XX/16), Latency, FPS."),
        ("State Progression Flow", "NORMAL -> DROWSINESS DETECTED -> WARNING -> RECOVERY CHECK -> RECOVERED / REST REQUIRED"),
        ("Presentation Demo Mode", "Press 'M' key to toggle Demo Simulation Mode for clean presentation testing without fake random model probability hacks."),
        ("Safety Disclaimer Banner", "Always displays 'Research Prototype — Not a Vehicle Control System' on lower panel.")
    ]
    for title, desc in feats:
        p = tf13.add_paragraph()
        p.text = f"• {title}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_GREEN
        p.space_before = Pt(10)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_PRIMARY
        
    set_speaker_notes(slide13,
                      "Our live demonstration presents a dual-panel dashboard featuring real-time probabilities, state banners, interactive recovery checks, and manual alert logging.",
                      "Functional software prototype demonstration.",
                      "Is this connected to vehicle control hardware?")

    # =========================================================
    # SLIDE 14: LIMITATIONS + FUTURE WORK
    # =========================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14)
    add_header(slide14, "Research Limitations & Future Directions")
    add_footer(slide14)
    
    # Left: Limitations
    l_box = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    l_box.fill.solid()
    l_box.fill.fore_color.rgb = CARD_BG
    l_box.line.color.rgb = ACCENT_RED
    tf_l = l_box.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "DOCUMENTED LIMITATIONS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED
    
    lims = [
        "Cross-domain shift degrades UTA-RLDD ROC-AUC (0.5425)",
        "High UTA-RLDD false positive rate (69.02% at t=0.5)",
        "Raw webcam probability saturation near 1.0",
        "Participant-specific feature inversion on P07",
        "Software prototype (no vehicle control/CAN bus connection)",
        "No clinical wakefulness or medical diagnosis claimed"
    ]
    for lim in lims:
        p = tf_l.add_paragraph()
        p.text = "• " + lim
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(8)
        
    # Right: Future Work
    f_box = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.6), Inches(4.8))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = CARD_BG
    f_box.line.color.rgb = ACCENT_GREEN
    tf_f = f_box.text_frame
    tf_f.word_wrap = True
    
    p = tf_f.paragraphs[0]
    p.text = "FUTURE RESEARCH DIRECTIONS"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    
    futs = [
        "Multi-source training on larger diverse driving datasets",
        "Subject-independent Leave-One-Group-Out benchmarking",
        "Domain-Adversarial Neural Networks (DANN) feature alignment",
        "Dynamic local adaptive illumination normalization (CLAHE)",
        "Deep probability fusion with Eye Aspect Ratio (EAR) landmarks",
        "Edge hardware deployment (NVIDIA Jetson Nano) with IR camera"
    ]
    for fut in futs:
        p = tf_f.add_paragraph()
        p.text = "• " + fut
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_PRIMARY
        p.space_before = Pt(8)
        
    set_speaker_notes(slide14,
                      "We explicitly document 10 operational limitations and outline future work including domain-adversarial networks and multi-modal landmark fusion.",
                      "Honest research disclosure and future roadmap.",
                      "How would you reduce false positive rates in future work?")

    # =========================================================
    # SLIDE 15: CONCLUSION & Q&A
    # =========================================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15)
    
    card15 = slide15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.1))
    card15.fill.solid()
    card15.fill.fore_color.rgb = CARD_BG
    card15.line.color.rgb = ACCENT_BLUE
    card15.line.width = Pt(2)
    
    tf15 = card15.text_frame
    tf15.word_wrap = True
    
    p = tf15.paragraphs[0]
    p.text = "DETECTION IS ONLY THE FIRST STEP"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    p2 = tf15.add_paragraph()
    p2.text = "MobileNetV2 + LSTM Architecture  +  Real-Time Monitoring\n+  Closed-Loop Intervention  +  Recovery Verification"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_GREEN
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(20)
    
    p3 = tf15.add_paragraph()
    p3.text = "The research prototype demonstrates an end-to-end safety-oriented workflow while explicitly identifying cross-domain generalization as a key limitation and future research direction."
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_PRIMARY
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(25)
    
    p4 = tf15.add_paragraph()
    p4.text = "THANK YOU!\nQUESTIONS & ANSWERS"
    p4.font.size = Pt(22)
    p4.font.bold = True
    p4.font.color.rgb = ACCENT_ORANGE
    p4.alignment = PP_ALIGN.CENTER
    p4.space_before = Pt(35)
    
    set_speaker_notes(slide15,
                      "To conclude, detection alone is insufficient; our prototype closes the loop by intervening and verifying recovery before resuming monitoring. Thank you.",
                      "End-to-end closed-loop driver safety monitoring system.",
                      "What is the single most important takeaway from your project?")

    # Save Presentation
    prs.save(PPTX_PATH)
    print(f"\n==================================================")
    print(f"POWERPOINT PRESENTATION GENERATED SUCCESSFULLY!")
    print(f"Saved To: {PPTX_PATH}")
    print(f"Slide Count: {len(prs.slides)}")
    print(f"==================================================\n")

if __name__ == "__main__":
    create_presentation()
