"""
Signature Detection — Accuracy Evaluator
-----------------------------------------
Compare model predictions vs your manually verified ground truth.
Reports Accuracy, Precision, Recall, F1, Specificity and lists every mistake
with a suggested fix.

Usage: python signature_accuracy.py
Edit PDF_PATH and GROUND_TRUTH_SIGNED below.
"""

import cv2
import numpy as np
import os
import warnings
from pathlib import Path

try:
    from skimage import measure, morphology
    from skimage.measure import regionprops
except ImportError:
    os.system("pip install scikit-image -q")
    from skimage import measure, morphology
    from skimage.measure import regionprops

try:
    import fitz
    if not hasattr(fitz, "open"):
        raise ImportError
except ImportError:
    os.system("pip uninstall fitz frontend PyMuPDF -y -q")
    os.system("pip install PyMuPDF -q")
    import fitz


# ════════════════════════════════════════════════════════════════════════════
# !! EDIT THIS SECTION !!
# ════════════════════════════════════════════════════════════════════════════

# sample_2
PDF_PATH = r"D:\HC\output_sample_2.pdf"
DPI      = 150
DEBUG    = False

# ## withou xerox copies
GROUND_TRUTH_SIGNED = {
    5, 6, 10, 11, 12, 29, 30
}
## entire doc with xerox copies
# PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\sample_1.pdf"
# GROUND_TRUTH_SIGNED = {
#     1, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 32, 33, 41, 42, 43, 54, 55, 59
# }



# ## sample_3
# PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_sample_3.pdf"
# DPI      = 150
# DEBUG    = False

# ## withou xerox copies
# GROUND_TRUTH_SIGNED = {
#     5, 6, 7, 8, 12, 13, 17, 20, 35, 36
# }

# # # entire doc with xerox copies
# # PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\sample_1.pdf"
# # GROUND_TRUTH_SIGNED = {
# #     1, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 32, 33, 41, 42, 43, 54, 55, 59
# # }


# ## sample_1
# PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_sample_1.pdf"
# DPI      = 150
# DEBUG    = False

# ## withou xerox copies
# GROUND_TRUTH_SIGNED = {
#     1, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 32
# }
# ## entire doc with xerox copies
# # PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\sample_1.pdf"
# # GROUND_TRUTH_SIGNED = {
# #     1, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 32, 33, 41, 42, 43, 54, 55, 59
# # }


# ## sample_0
# PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_part1.pdf"
# # PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\CS_COM_ed_AWB20240000735D202500006_ocr.pdf"
# DPI      = 150
# DEBUG    = False

# ## without supporting documents
# GROUND_TRUTH_SIGNED = {
#     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#     21, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
#     43, 44, 45, 46, 47, 48, 49, 50
# }

# ## with all the supporting documents
# # GROUND_TRUTH_SIGNED = {
# #     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
# #     21, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
# #     43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 164, 165, 177, 178, 179, 186, 187, 199, 200
# # }

# ════════════════════════════════════════════════════════════════════════════

# Must match signature_detector.py constants
CONSTANT_1          = 84
CONSTANT_2          = 250
CONSTANT_3          = 100
CONSTANT_4          = 18
MAX_AVERAGE_AREA    = 100
MAX_BIGGEST_BLOB    = 9000
MIN_ECCENTRICITY    = 0.5
MIN_ECCENTRIC_RATIO = 0.4
MIN_SURVIVING_PIXELS = 500   # new gaurd= new constant — ignore pages with too little ink
# new small_threshold = ((61/84) * 150) + 30 = 109 + 30 = 139px

def page_to_image(page, dpi=200):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def detect_signature(gray_img):
    _, binary    = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    blobs        = binary > binary.mean()
    blobs_labels = measure.label(blobs, background=1)

    total_area, counter, biggest = 0, 0, 0
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter    += 1
        if region.area >= 150 and region.area > biggest:
            biggest = region.area

    if counter == 0:
        return False, "blank page"

    average = total_area / counter

    if average > MAX_AVERAGE_AREA:
        return False, f"avg area {average:.1f} > {MAX_AVERAGE_AREA}"
    if biggest > MAX_BIGGEST_BLOB:
        return False, f"biggest blob {biggest} > {MAX_BIGGEST_BLOB}"

    small_threshold = ((average / CONSTANT_1) * CONSTANT_2) + CONSTANT_3
    big_threshold   = small_threshold * CONSTANT_4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = morphology.remove_small_objects(blobs_labels, small_threshold)

    component_sizes       = np.bincount(cleaned.ravel())
    too_big_mask          = (component_sizes > big_threshold)[cleaned]
    cleaned[too_big_mask] = 0

    if np.count_nonzero(cleaned) == 0:
        return False, "no blobs survived size filter"

    cleaned_labels  = measure.label(cleaned > 0)
    eccentric_count = 0
    total_count     = 0
    for region in regionprops(cleaned_labels):
        if region.area < 50:
            continue
        total_count += 1
        if region.eccentricity > MIN_ECCENTRICITY:
            eccentric_count += 1

    if total_count == 0:
        return False, "no qualifying blobs"

    eccentric_ratio = eccentric_count / total_count
    if eccentric_ratio < MIN_ECCENTRIC_RATIO:
        return False, f"eccentric_ratio {eccentric_ratio:.2f} < {MIN_ECCENTRIC_RATIO} (round = stamp)"

    return True, f"eccentric_ratio {eccentric_ratio:.2f}"

# PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_sample_1.pdf"
def evaluate():
    pdf_path   = Path(PDF_PATH)
    doc        = fitz.open(str(pdf_path))
    total      = len(doc)
    eval_up_to = max(GROUND_TRUTH_SIGNED) if GROUND_TRUTH_SIGNED else total

    print(f"\n  PDF            : {pdf_path.name}")
    print(f"  Total pages    : {total}")
    print(f"  Evaluating     : pages 1 – {eval_up_to}")
    print(f"  GT signed pages: {sorted(GROUND_TRUTH_SIGNED)}")
    print(f"  {'─'*65}")
    print(f"  {'Page':<6} {'GT':^10} {'Predicted':^12} {'Result':^20}  Reason")
    print(f"  {'─'*65}")

    TP = FP = TN = FN = 0
    mistakes = []

    for page_num in range(eval_up_to):
        human_num    = page_num + 1
        gray_img     = page_to_image(doc[page_num], dpi=DPI)
        pred, reason = detect_signature(gray_img)
        gt           = human_num in GROUND_TRUTH_SIGNED

        if   gt and pred:      outcome = "TP  ✓";               TP += 1
        elif not gt and pred:  outcome = "FP  ✗  FALSE POSITIVE"; FP += 1; mistakes.append((human_num, "FALSE POSITIVE", reason))
        elif gt and not pred:  outcome = "FN  ✗  FALSE NEGATIVE"; FN += 1; mistakes.append((human_num, "FALSE NEGATIVE", reason))
        else:                  outcome = "TN  ✓";               TN += 1

        gt_str   = "SIGNED"   if gt   else "unsigned"
        pred_str = "SIGNED"   if pred else "unsigned"
        print(f"  {human_num:<6} {gt_str:^10} {pred_str:^12} {outcome:<22}  {reason}")

    doc.close()

    total_eval  = TP + FP + TN + FN
    accuracy    = (TP + TN) / total_eval * 100 if total_eval   else 0
    precision   = TP / (TP + FP)         * 100 if (TP + FP)   else 0
    recall      = TP / (TP + FN)         * 100 if (TP + FN)   else 0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    specificity = TN / (TN + FP)         * 100 if (TN + FP)   else 0

    # print(f" sample_0 :- ")
    print(f"\n  {'='*60}")
    print(f"  ACCURACY REPORT  (evaluated {total_eval} pages)")
    print(f"  {'='*60}")
    print(f"  Correct   : {TP + TN} / {total_eval}")
    print(f"  Wrong     : {FP + FN} / {total_eval}")
    print(f"  {'─'*60}")
    print(f"  True  Positives (TP) : {TP:>4}   correctly found signatures")
    print(f"  True  Negatives (TN) : {TN:>4}   correctly found no signature")
    print(f"  False Positives (FP) : {FP:>4}   said SIGNED, was unsigned")
    print(f"  False Negatives (FN) : {FN:>4}   said unsigned, was SIGNED")
    print(f"  {'─'*60}")
    print(f"  Accuracy             : {accuracy:.1f}%")
    print(f"  Precision            : {precision:.1f}%   (predicted signed → actually signed)")
    print(f"  Recall               : {recall:.1f}%   (all real signatures → detected)")
    print(f"  F1 Score             : {f1:.1f}%   (main overall metric)")
    print(f"  Specificity          : {specificity:.1f}%   (unsigned pages correctly rejected)")
    print(f"  {'='*60}")

    if mistakes:
        print(f"\n  MISTAKES ({len(mistakes)}) :")
        for page, mtype, reason in mistakes:
            print(f"    Page {page:>3}  {mtype}  |  {reason}")
        print(f"\n  HOW TO FIX:")
        if any(m[1] == "FALSE POSITIVE" for m in mistakes):
            print(f"    False Positives -> Lower MIN_ECCENTRIC_RATIO (now {MIN_ECCENTRIC_RATIO})")
            print(f"                    -> Or lower MAX_BIGGEST_BLOB  (now {MAX_BIGGEST_BLOB})")
        if any(m[1] == "FALSE NEGATIVE" for m in mistakes):
            print(f"    False Negatives -> Raise MAX_AVERAGE_AREA     (now {MAX_AVERAGE_AREA})")
            print(f"                    -> Or raise MAX_BIGGEST_BLOB  (now {MAX_BIGGEST_BLOB})")
            print(f"                    -> Or lower MIN_ECCENTRIC_RATIO (now {MIN_ECCENTRIC_RATIO})")
    else:
        print(f"\n  No mistakes! Perfect accuracy on evaluated pages.")

    print()
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    evaluate()
