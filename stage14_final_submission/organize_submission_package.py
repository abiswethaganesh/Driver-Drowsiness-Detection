"""
STAGE 14 — SUBMISSION PACKAGE ORGANIZER SCRIPT
Copies and structures all report markdown files, figures, demo scripts, viva Q&A, and audit logs
into the final submission directory layout.
"""

import os
import shutil

BASE_DIR = r"d:\Sem 7\NNDL\Project Demo"
STAGE12_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage12_final_demo")
STAGE13_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage13_submission")
STAGE14_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage14_final_submission")

def organize_package():
    print("==================================================")
    print("ORGANIZING STAGE 14 FINAL SUBMISSION PACKAGE")
    print("==================================================")
    
    # Define subdirectories
    subdirs = ["project_report", "documentation", "demo", "figures", "viva", "audits"]
    for sd in subdirs:
        os.makedirs(os.path.join(STAGE14_DIR, sd), exist_ok=True)
        
    # Copy Project Report
    src_report = os.path.join(STAGE13_DIR, "final_project_report.md")
    dst_report = os.path.join(STAGE14_DIR, r"project_report\final_project_report.md")
    if os.path.exists(src_report):
        shutil.copy2(src_report, dst_report)
        print(f"Copied: {dst_report}")
        
    # Copy Documentation
    doc_files = ["methodology_table.md", "results_table.md"]
    for df in doc_files:
        src = os.path.join(STAGE13_DIR, df)
        dst = os.path.join(STAGE14_DIR, "documentation", df)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")
            
    doc_files_12 = ["limitations.md", "dataset_summary.md"]
    for df in doc_files_12:
        src = os.path.join(STAGE12_DIR, df)
        dst = os.path.join(STAGE14_DIR, "documentation", df)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")
            
    # Copy Demo
    demo_files_12 = ["final_demo.py"]
    for df in demo_files_12:
        src = os.path.join(STAGE12_DIR, df)
        dst = os.path.join(STAGE14_DIR, "demo", df)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")
            
    demo_files_13 = ["demo_script.md", "demo_checklist.md"]
    for df in demo_files_13:
        src = os.path.join(STAGE13_DIR, df)
        dst = os.path.join(STAGE14_DIR, "demo", df)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")
            
    # Copy Figures
    fig_dir_12 = os.path.join(STAGE12_DIR, "figures")
    if os.path.exists(fig_dir_12):
        for fig_name in os.listdir(fig_dir_12):
            if fig_name.endswith(".png"):
                src = os.path.join(fig_dir_12, fig_name)
                dst = os.path.join(STAGE14_DIR, "figures", fig_name)
                shutil.copy2(src, dst)
                print(f"Copied Figure: {dst}")
                
    # Copy Viva
    viva_files = ["viva_defense.md", "elevator_pitch.md"]
    for vf in viva_files:
        src = os.path.join(STAGE13_DIR, vf)
        dst = os.path.join(STAGE14_DIR, "viva", vf)
        if os.path.exists(src):
            shutil.copy2(src, vf_dst if 'vf_dst' in locals() else dst)
            print(f"Copied: {dst}")
            
    # Copy Audits
    audit_files_13 = ["consistency_audit.md", "submission_audit.json"]
    for af in audit_files_13:
        src = os.path.join(STAGE13_DIR, af)
        dst = os.path.join(STAGE14_DIR, "audits", af)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied Audit: {dst}")
            
    src_ppt_audit = os.path.join(STAGE14_DIR, "ppt_quality_report.json")
    dst_ppt_audit = os.path.join(STAGE14_DIR, r"audits\ppt_quality_report.json")
    if os.path.exists(src_ppt_audit) and src_ppt_audit != dst_ppt_audit:
        shutil.copy2(src_ppt_audit, dst_ppt_audit)
        print(f"Copied Audit: {dst_ppt_audit}")

    print("\nSUBMISSION PACKAGE STRUCTURED SUCCESSFULLY!\n")

if __name__ == "__main__":
    organize_package()
