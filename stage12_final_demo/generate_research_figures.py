"""
STAGE 12 — RESEARCH FIGURES GENERATOR
Generates 8 presentation-quality scientific charts and diagrams for the final report and presentation slides.
Uses existing empirical recorded metrics without model retraining.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BASE_DIR = r"d:\Sem 7\NNDL\Project Demo"
FIG_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage12_final_demo\figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# Figure 1: System Architecture Pipeline
# -------------------------------------------------------------
def generate_fig1():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')
    
    steps = [
        "Webcam / Video\nInput",
        "Face & Landmark\nDetection",
        "128×128 Face\nCrop & Normalization",
        "16-Frame Temporal\nSequence Buffer",
        "TimeDistributed\nMobileNetV2 (Frozen)",
        "64-Unit LSTM\nTemporal Head",
        "Drowsiness Probability\n& Closed-Loop System"
    ]
    
    n_steps = len(steps)
    box_w = 0.11
    box_h = 0.55
    spacing = (1.0 - n_steps * box_w) / (n_steps + 1)
    
    for i, step in enumerate(steps):
        x = spacing + i * (box_w + spacing)
        y = 0.22
        
        # Color coding
        color = '#1f77b4' if i < 4 else ('#ff7f0e' if i < 6 else '#2ca02c')
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                       linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h/2, step, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#111111')
        
        if i < n_steps - 1:
            ax.annotate('', xy=(x + box_w + spacing*0.8, y + box_h/2), xytext=(x + box_w + 0.005, y + box_h/2),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
                        
    plt.title("FIGURE 1: Driver Safety Monitoring System Architecture Pipeline", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_system_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 1: {out_path}")

# -------------------------------------------------------------
# Figure 2: Closed-Loop Intervention Workflow
# -------------------------------------------------------------
def generate_fig2():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.axis('off')
    
    nodes = {
        "NORMAL": (0.2, 0.75, "#2ca02c", "NORMAL\nMonitoring Driver"),
        "DROWSY": (0.5, 0.75, "#d62728", "DROWSINESS_DETECTED\n(Smoothed Prob ≥ 0.50)"),
        "WARN": (0.8, 0.75, "#ff7f0e", "WARNING\nVisual/Audible Alert"),
        "RECOVERY": (0.5, 0.35, "#1f77b4", "RECOVERY_CHECK\nPrompt: Look at Camera\n& Keep Eyes Open"),
        "REST": (0.8, 0.35, "#9467bd", "REST_REQUIRED\nSimulated 3-Min Rest\nProtocol (03:00)"),
        "RESUME": (0.2, 0.35, "#2ca02c", "RECOVERY_VERIFIED\nResume Normal\nMonitoring")
    }
    
    for key, (x, y, color, label) in nodes.items():
        rect = patches.FancyBboxPatch((x-0.1, y-0.1), 0.2, 0.2, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#111111')
        
    # Arrows
    ax.annotate('', xy=(0.38, 0.75), xytext=(0.31, 0.75), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
    ax.annotate('', xy=(0.68, 0.75), xytext=(0.61, 0.75), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
    ax.annotate('', xy=(0.5, 0.47), xytext=(0.8, 0.63), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
    ax.annotate('', xy=(0.68, 0.35), xytext=(0.62, 0.35), arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, ls="--"))
    ax.annotate('', xy=(0.31, 0.35), xytext=(0.38, 0.35), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))
    ax.annotate('', xy=(0.2, 0.63), xytext=(0.2, 0.47), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))
    
    plt.title("FIGURE 2: Novel Closed-Loop Intervention & Recovery Verification Workflow", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_closed_loop_workflow.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 2: {out_path}")

# -------------------------------------------------------------
# Figure 3: Dataset Distribution
# -------------------------------------------------------------
def generate_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    
    # NITYMED Composition
    nm_labels = ['NITYMED Drowsy\n(Label 1: 100%)']
    nm_counts = [5822]
    ax1.bar(nm_labels, nm_counts, color='#d62728', width=0.4, alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Sequence Count", fontsize=9, fontweight='bold')
    ax1.set_title("NITYMED Dataset Composition (5,822 Seqs)", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(nm_counts):
        ax1.text(i, v + 100, f"{v:,}\n(100% Drowsy)", ha='center', fontsize=8, fontweight='bold')
    ax1.set_ylim(0, 7000)
    
    # UTA-RLDD Composition
    rldd_labels = ['Alert\n(Class 0)', 'Low Vigilant\n(Class 5)', 'Drowsy\n(Class 10)']
    rldd_counts = [7039, 7106, 5864]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax2.bar(rldd_labels, rldd_counts, color=colors, width=0.5, alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Sequence Count", fontsize=9, fontweight='bold')
    ax2.set_title("UTA-RLDD Local Subset Composition (20,009 Seqs)", fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars, rldd_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 150, f"{v:,}", ha='center', fontsize=8, fontweight='bold')
    ax2.set_ylim(0, 8500)
    
    plt.suptitle("FIGURE 3: Training & Validation Dataset Class Distributions", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig3_dataset_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 3: {out_path}")

# -------------------------------------------------------------
# Figure 4: Model Architecture Block Diagram
# -------------------------------------------------------------
def generate_fig4():
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.axis('off')
    
    layers = [
        ("Input Sequence\n(16, 128, 128, 3)", "#7f7f7f"),
        ("TimeDistributed\nMobileNetV2 (Frozen)", "#1f77b4"),
        ("Global Average\nPooling (1280-d)", "#aec7e8"),
        ("LSTM Layer\n(64 Units)", "#ff7f0e"),
        ("Dropout\n(p = 0.3)", "#ffbb78"),
        ("Dense Layer\n(32 Units, ReLU)", "#2ca02c"),
        ("Dense Output\n(1 Unit, Sigmoid)", "#d62728")
    ]
    
    x_positions = np.linspace(0.08, 0.92, len(layers))
    for i, (name, col) in enumerate(layers):
        x = x_positions[i]
        rect = patches.FancyBboxPatch((x-0.055, 0.3), 0.11, 0.4, boxstyle="round,pad=0.01,rounding_size=0.02",
                                       linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.2)
        ax.add_patch(rect)
        ax.text(x, 0.5, name, ha='center', va='center', fontsize=7, fontweight='bold', color='#111111')
        if i < len(layers) - 1:
            next_x = x_positions[i+1]
            ax.annotate('', xy=(next_x-0.058, 0.5), xytext=(x+0.058, 0.5), arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
            
    plt.title("FIGURE 4: Deep Learning Model Architecture (MobileNetV2 + 64-Unit LSTM)", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_model_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 4: {out_path}")

# -------------------------------------------------------------
# Figure 5: Validation vs External Test Performance
# -------------------------------------------------------------
def generate_fig5():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    categories = ['Validation Set\n(Combined)', 'Combined Test\n(NITYMED + UTA)', 'External UTA-RLDD\nTest Set']
    auc_values = [0.947850, 0.638542, 0.542460]
    f1_values = [0.965200, 0.824284, 0.765152]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, auc_values, width, label='ROC-AUC', color='#1f77b4', alpha=0.85, edgecolor='black')
    rects2 = ax.bar(x + width/2, f1_values, width, label='F1-Score', color='#2ca02c', alpha=0.85, edgecolor='black')
    
    ax.set_ylabel('Metric Value (0.0 to 1.0)', fontsize=9, fontweight='bold')
    ax.set_title('FIGURE 5: Baseline Model Validation vs. External Held-Out Test Performance', fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.15)
    
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    # Annotate domain gap
    ax.annotate('External Domain Gap', xy=(2, 0.55), xytext=(1.2, 0.45),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                fontsize=8, fontweight='bold', color='red')
                
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig5_validation_vs_external_test.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 5: {out_path}")

# -------------------------------------------------------------
# Figure 6: Domain Shift Evidence (Luminance Shift)
# -------------------------------------------------------------
def generate_fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    
    groups = ['NITYMED Drowsy', 'UTA-RLDD Alert', 'UTA-RLDD Drowsy']
    lum_means = [69.60, 116.34, 107.61]
    lum_stds = [28.79, 70.53, 67.68]
    
    bars1 = ax1.bar(groups, lum_means, yerr=lum_stds, capsize=5, color=['#d62728', '#2ca02c', '#ff7f0e'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Mean Luminance (0 to 255)", fontsize=9, fontweight='bold')
    ax1.set_title("Pixel Luminance Shift across Sources", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, m in zip(bars1, lum_means):
        ax1.text(bar.get_x() + bar.get_width()/2, m + 12, f"{m:.1f}", ha='center', fontsize=8, fontweight='bold')
    ax1.set_ylim(0, 200)
    
    # False Positive Rate (FPR) vs Domain Shift
    fpr_groups = ['Baseline (A1)', 'Stage 10B (Jitter)', 'Stage 10C (B2 Norm)']
    fpr_vals = [69.02, 60.30, 25.40]
    bars2 = ax2.bar(fpr_groups, fpr_vals, color=['#d62728', '#ff7f0e', '#1f77b4'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel("External UTA-RLDD False Positive Rate (%)", fontsize=9, fontweight='bold')
    ax2.set_title("UTA-RLDD False Positive Rate Reduction", fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars2, fpr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.2f}%", ha='center', fontsize=8, fontweight='bold')
    ax2.set_ylim(0, 85)
    
    plt.suptitle("FIGURE 6: Quantitative Evidence of Illumination Domain Shift", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig6_domain_shift_evidence.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 6: {out_path}")

# -------------------------------------------------------------
# Figure 7: Experiment Comparison across Stages
# -------------------------------------------------------------
def generate_fig7():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    experiments = ['Stage 08\nBaseline', 'Stage 08E\nFine-Tuning', 'Stage 10B\nDomain-Aware', 'Stage 10C\nModel B2']
    val_auc = [0.94785, 0.87517, 0.91140, 0.68500]
    uta_auc = [0.54246, 0.52135, 0.50980, 0.50500]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    r1 = ax.bar(x - width/2, val_auc, width, label='Validation ROC-AUC', color='#1f77b4', alpha=0.85, edgecolor='black')
    r2 = ax.bar(x + width/2, uta_auc, width, label='UTA-RLDD Test ROC-AUC', color='#ff7f0e', alpha=0.85, edgecolor='black')
    
    ax.set_ylabel('ROC-AUC Score (0.0 to 1.0)', fontsize=9, fontweight='bold')
    ax.set_title('FIGURE 7: Systematic Experiment Comparison across Development Stages', fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.15)
    
    for r in r1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig7_experiment_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 7: {out_path}")

# -------------------------------------------------------------
# Figure 8: Real-Time Inference Performance
# -------------------------------------------------------------
def generate_fig8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=300)
    
    # Latency Metrics
    lat_labels = ['Mean Latency', 'Median Latency', 'P95 Latency']
    lat_values = [78.65, 74.32, 108.86]
    bars1 = ax1.bar(lat_labels, lat_values, color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.85, edgecolor='black', width=0.5)
    ax1.set_ylabel("Inference Latency per 16-Frame Seq (ms)", fontsize=9, fontweight='bold')
    ax1.set_title("Real-Time Webcam Sequence Latency", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars1, lat_values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 3, f"{v:.2f} ms", ha='center', fontsize=8, fontweight='bold')
    ax1.set_ylim(0, 135)
    
    # FPS Distribution
    fps_labels = ['Mean FPS']
    fps_values = [15.73]
    bars2 = ax2.bar(fps_labels, fps_values, color='#2ca02c', alpha=0.85, edgecolor='black', width=0.3)
    ax2.set_ylabel("Frames Per Second (FPS)", fontsize=9, fontweight='bold')
    ax2.set_title("Real-Time Webcam Throughput", fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars2, fps_values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.2f} FPS", ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 25)
    
    plt.suptitle("FIGURE 8: Real-Time Computational Efficiency & Latency Profile", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig8_realtime_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 8: {out_path}")

def generate_all():
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    print("\nALL 8 RESEARCH FIGURES GENERATED SUCCESSFULLY!\n")

if __name__ == "__main__":
    generate_all()
