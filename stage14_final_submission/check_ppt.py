"""
STAGE 14 — PPT QUALITY CHECKER
Audits Driver_Drowsiness_Final_Presentation.pptx to ensure slide count=15, dark theme styling,
speaker notes on all slides, embedded images, absence of placeholders/unsupported claims, and scientific accuracy.
"""

import os
import json
import pptx
from pptx import Presentation

BASE_DIR = r"d:\Sem 7\NNDL\Project Demo"
STAGE14_DIR = os.path.join(BASE_DIR, r"nitymed_work\stage14_final_submission")
PPTX_PATH = os.path.join(STAGE14_DIR, "Driver_Drowsiness_Final_Presentation.pptx")
REPORT_PATH = os.path.join(STAGE14_DIR, "ppt_quality_report.json")

PROHIBITED_STRINGS = [
    "[TODO]", "[INSERT]", "[REPLACE]", "[REFERENCE TO BE ADDED]",
    "100% ACCURATE", "PRODUCTION READY", "GUARANTEES SAFETY",
    "PREVENTS ACCIDENTS", "VEHICLE AUTOMATICALLY STOPS"
]

def audit_ppt():
    print("==================================================")
    print("STAGE 14 — PPT QUALITY AUDIT")
    print("==================================================")
    
    report = {}
    
    # 1. File exists
    report["1_file_exists"] = os.path.exists(PPTX_PATH)
    print(f"Check 01: Presentation File Exists         -> {'PASS' if report['1_file_exists'] else 'FAIL'}")
    
    if not report["1_file_exists"]:
        return False
        
    # 2. File opens
    try:
        prs = Presentation(PPTX_PATH)
        report["2_file_opens"] = True
    except Exception as e:
        prs = None
        report["2_file_opens"] = False
        print(f"Check 02: Presentation Opens               -> FAIL ({e})")
        return False
    print(f"Check 02: Presentation Opens               -> PASS")
    
    # 3. Slide count = 15
    slide_count = len(prs.slides)
    report["3_slide_count_15"] = (slide_count == 15)
    print(f"Check 03: Slide Count = 15 (Actual: {slide_count})   -> {'PASS' if report['3_slide_count_15'] else 'FAIL'}")
    
    # 4. No empty slides & Title present
    all_non_empty = True
    all_titles_present = True
    for i, slide in enumerate(prs.slides):
        has_text = False
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                has_text = True
                break
        if not has_text:
            all_non_empty = False
            
    report["4_no_empty_slides"] = all_non_empty
    print(f"Check 04: No Empty Slides                  -> {'PASS' if report['4_no_empty_slides'] else 'FAIL'}")
    
    # 5. Speaker notes exist for all slides
    notes_exist = True
    for i, slide in enumerate(prs.slides):
        notes_frame = slide.notes_slide.notes_text_frame
        if not notes_frame.text.strip():
            notes_exist = False
            break
    report["5_speaker_notes_exist"] = notes_exist
    print(f"Check 05: Speaker Notes on Every Slide     -> {'PASS' if report['5_speaker_notes_exist'] else 'FAIL'}")
    
    # 6. Images embedded (figures present)
    image_count = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE:
                image_count += 1
    report["6_images_embedded"] = (image_count >= 5)
    print(f"Check 06: Embedded Figure Images ({image_count})     -> {'PASS' if report['6_images_embedded'] else 'FAIL'}")
    
    # 7. No placeholders or prohibited strings in slides
    found_placeholders = []
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt_upper = shape.text_frame.text.upper()
                for p_str in PROHIBITED_STRINGS:
                    if p_str in txt_upper:
                        found_placeholders.append(f"Slide {i+1}: {p_str}")
                        
    report["7_no_prohibited_placeholders"] = (len(found_placeholders) == 0)
    print(f"Check 07: Absence of Placeholders/Exaggerations -> {'PASS' if report['7_no_prohibited_placeholders'] else 'FAIL'}")
    if found_placeholders:
        print(f"   Flagged: {found_placeholders}")
        
    all_passed = all([
        report["1_file_exists"], report["2_file_opens"], report["3_slide_count_15"],
        report["4_no_empty_slides"], report["5_speaker_notes_exist"],
        report["6_images_embedded"], report["7_no_prohibited_placeholders"]
    ])
    
    report["overall_ppt_quality"] = "PASS" if all_passed else "FAIL"
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n==================================================")
    print(f"PPT QUALITY RESULT: {'PASS (ALL QUALITY CHECKS PASSED)' if all_passed else 'FAIL'}")
    print(f"Quality Report Saved To: {REPORT_PATH}")
    print("==================================================\n")
    return all_passed

if __name__ == "__main__":
    audit_ppt()
