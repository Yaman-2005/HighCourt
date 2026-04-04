"""
Signature Detection - DEBUG / Parameter Tuning Tool
-----------------------------------------------------
Renders each requested page with colour-coded blobs so you can see
exactly what the algorithm is keeping vs discarding.

  GREEN      = KEPT  — algorithm thinks this is a signature stroke
  RED        = REMOVED — too big (text block / table / stamp image)
  LIGHT GREY = REMOVED — too small (noise / dust)

Output images are saved to:  debug_output/page_XXX_debug.png

Usage: python signature_debug.py
Edit PDF_PATH and PAGE_NUMBERS below.
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


# ── EDIT THESE ───────────────────────────────────────────────────────────────
## for sample 0
# PDF_PATH     = r"C:\Users\User\Documents\internship\high_court\working_code\data\CS_COM_ed_AWB20240000735D202500006.pdf"
# PAGE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]   # pages to debug (1-based), or set to "all"

# PDF_PATH     = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_part1.pdf"
# PAGE_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]   # pages to debug (1-based), or set to "all"


## sample_1
# PDF_PATH     = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_sample_1.pdf"

# PAGE_NUMBERS =  list(range(1, 33))  # pages to debug (1-based), or set to "all"
# DPI          = 150

# ## sample_2
PDF_PATH     = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_sample_2.pdf"

PAGE_NUMBERS =  list(range(1, 37))  # pages to debug (1-based), or set to "all"
DPI          = 150

# ── Tunable constants — must match signature_detector.py ─────────────────────
CONSTANT_1          = 84
CONSTANT_2          = 250
CONSTANT_3          = 100
CONSTANT_4          = 18
MAX_AVERAGE_AREA    = 100
MAX_BIGGEST_BLOB    = 9000
MIN_ECCENTRICITY    = 0.5
MIN_ECCENTRIC_RATIO = 0.4
MIN_SURVIVING_PIXELS = 500
dpi=200
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("debug_output/sample_1")
OUTPUT_DIR.mkdir(exist_ok=True)


def page_to_image(page, dpi=200):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def debug_page(gray_img, page_num):
    print(f"\n  Page {page_num}:")

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
        print(f"    BLANK PAGE — no components found")
        return False

    average         = total_area / counter
    small_threshold = ((average / CONSTANT_1) * CONSTANT_2) + CONSTANT_3
    big_threshold   = small_threshold * CONSTANT_4

    print(f"    average blob area   : {average:.1f} px")
    print(f"    biggest blob        : {biggest} px")
    print(f"    small_threshold     : {small_threshold:.1f}  (blobs BELOW this = noise, removed)")
    print(f"    big_threshold       : {big_threshold:.1f}  (blobs ABOVE this = text/table, removed)")
    print(f"    => kept blobs are between {small_threshold:.0f} and {big_threshold:.0f} px")

    # Early-exit guards (same as detector)
    if average > MAX_AVERAGE_AREA:
        print(f"    !! GUARD 1 triggered: avg area {average:.1f} > {MAX_AVERAGE_AREA} — would be rejected")
    if biggest > MAX_BIGGEST_BLOB:
        print(f"    !! GUARD 2 triggered: biggest blob {biggest} > {MAX_BIGGEST_BLOB} — would be rejected")

    # ── Build colour debug image ─────────────────────────────────────────────
    h, w  = gray_img.shape
    vis   = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    for region in regionprops(blobs_labels):
        coords = region.coords
        if region.area < small_threshold:
            color = (200, 200, 200)   # light grey — too small (noise)
        elif region.area > big_threshold:
            color = (0, 0, 255)       # RED (BGR) — too big (text/table/stamp)
        else:
            color = (0, 200, 0)       # GREEN — KEPT (candidate signature blob)
        vis[coords[:, 0], coords[:, 1]] = color

    # Overlay original image faintly
    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(gray_rgb, 0.25, vis, 0.75, 0)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_y = 30
    for label, color, desc in [
        ("GREEN",      (0, 180, 0),    "KEPT — algorithm thinks this is a signature"),
        ("RED",        (0, 0, 220),    "REMOVED — too big (text / table / border)"),
        ("LIGHT GREY", (160, 160, 160),"REMOVED — too small (noise / dust)"),
    ]:
        cv2.rectangle(combined, (10, legend_y - 15), (35, legend_y + 5), color, -1)
        cv2.putText(combined, f"{label}: {desc}", (45, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        legend_y += 30

    # ── Eccentricity check on kept blobs ────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = morphology.remove_small_objects(blobs_labels.copy(), small_threshold)

    component_sizes       = np.bincount(cleaned.ravel())
    too_big_mask          = (component_sizes > big_threshold)[cleaned]
    cleaned[too_big_mask] = 0

    surviving = np.count_nonzero(cleaned)
    eccentric_count = total_count = 0

    if surviving > 0:
        cleaned_labels = measure.label(cleaned > 0)
        for region in regionprops(cleaned_labels):
            if region.area < 50:
                continue
            total_count += 1
            if region.eccentricity > MIN_ECCENTRICITY:
                eccentric_count += 1

    eccentric_ratio = (eccentric_count / total_count) if total_count > 0 else 0.0
    print(f"    surviving pixels    : {surviving}")
    print(f"    eccentric_ratio     : {eccentric_ratio:.2f}  (need >= {MIN_ECCENTRIC_RATIO} to be 'handwriting')")

    # Final verdict
    if average > MAX_AVERAGE_AREA:
        verdict = "NO SIGNATURE  [Guard 1: avg area too high]"
    elif biggest > MAX_BIGGEST_BLOB:
        verdict = "NO SIGNATURE  [Guard 2: biggest blob too large]"
    elif surviving == 0:
        verdict = "NO SIGNATURE  [no blobs survived size filter]"
    elif eccentric_ratio < MIN_ECCENTRIC_RATIO:
        verdict = f"NO SIGNATURE  [Guard 3: eccentric_ratio {eccentric_ratio:.2f} < {MIN_ECCENTRIC_RATIO} — round blobs = stamp]"
    else:
        verdict = "SIGNATURE FOUND"

    print(f"    Verdict             : {verdict}")

    out_path = OUTPUT_DIR / f"page_{page_num:03d}_debug.png"
    cv2.imwrite(str(out_path), combined)
    print(f"    Debug image saved  -> {out_path}")

    return "SIGNATURE FOUND" in verdict


def main():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc   = fitz.open(str(pdf_path))
    total = len(doc)
    pages = range(1, total + 1) if PAGE_NUMBERS == "all" else PAGE_NUMBERS

    print(f"\n  PDF    : {pdf_path.name}  ({total} pages)")
    print(f"  Debug  : pages {list(pages)}")
    print(f"  Output : {OUTPUT_DIR.resolve()}")
    print(f"\n  {'='*55}")
    print("  HOW TO READ THE DEBUG IMAGES:")
    print("    GREEN      = blobs the algorithm KEEPS as 'signature'")
    print("    RED        = blobs removed (too big = text/tables/stamps)")
    print("    LIGHT GREY = blobs removed (too small = noise/dust)")
    print(f"  {'='*55}")

    for p in pages:
        if p < 1 or p > total:
            print(f"\n  Page {p} out of range (doc has {total} pages), skipping.")
            continue
        gray_img = page_to_image(doc[p - 1], dpi=DPI)
        debug_page(gray_img, p)

    doc.close()

    print(f"\n  Done. Open images in: {OUTPUT_DIR.resolve()}")
    print("\n  TUNING GUIDE:")
    print("  If GREEN covers printed text  -> lower  MAX_AVERAGE_AREA or MIN_ECCENTRIC_RATIO")
    print("  If real signature is RED      -> raise  MAX_BIGGEST_BLOB or CONSTANT_4")
    print("  If real signature is GREY     -> lower  CONSTANT_1 or CONSTANT_3")
    print("  If stamp still shows as GREEN -> lower  MIN_ECCENTRIC_RATIO\n")


if __name__ == "__main__":
    main()