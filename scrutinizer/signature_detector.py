"""
Signature Detection in PDF Pages
----------------------------------
Detects HANDWRITTEN INK SIGNATURES only.
Circular stamps, logos, printed images, QR codes are ignored.

Usage: python signature_detector.py
Edit PDF_PATH, DPI, DEBUG in main() below.
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


# ── Detection Constants ──────────────────────────────────────────────────────
# Original connected-component size filter constants
CONSTANT_1 = 84
CONSTANT_2 = 250
CONSTANT_3 = 100
CONSTANT_4 = 18

# Guard 1: Reject pages where average blob area is too high
# Handwritten ink blobs average 30-70px. Dense print/logos average >100px.
MAX_AVERAGE_AREA = 100

# Guard 2: Reject pages with a single very large blob
# Large stamps/images produce blobs >5000px. Handwritten strokes do not.
MAX_BIGGEST_BLOB = 5000

# Guard 3: Eccentricity — rejects circular stamps, keeps elongated pen strokes
# Eccentricity near 1.0 = thin/elongated (handwriting)
# Eccentricity near 0.0 = round (rubber stamp / seal)
MIN_ECCENTRICITY    = 0.5   # a blob is "stroke-like" if eccentricity > this
MIN_ECCENTRIC_RATIO = 0.4   # at least 40% of blobs must be stroke-like


def page_to_image(page, dpi: int = 150) -> np.ndarray:
    """Render a PDF page to a grayscale numpy array."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def detect_signature(gray_img: np.ndarray, debug: bool = False) -> dict:
    """
    Detect handwritten ink signatures in a grayscale page image.
    Returns dict with keys:
        has_signature : bool
        stats         : dict with reason and measurements
    """
    # Binarise
    _, binary    = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    blobs        = binary > binary.mean()
    blobs_labels = measure.label(blobs, background=1)

    total_area, counter, biggest = 0, 0, 0
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter    += 1
        if region.area >= 250 and region.area > biggest:
            biggest = region.area

    if counter == 0:
        return {"has_signature": False, "stats": {"reason": "blank page"}}

    average = total_area / counter

    # ── Guard 1: average blob area ───────────────────────────────────────────
    if average > MAX_AVERAGE_AREA:
        return {"has_signature": False, "stats": {
            "reason": f"avg blob area {average:.1f} > {MAX_AVERAGE_AREA} — dense print/logo, not handwriting",
            "average_area": round(average, 2)}}

    # ── Guard 2: biggest single blob ─────────────────────────────────────────
    if biggest > MAX_BIGGEST_BLOB:
        return {"has_signature": False, "stats": {
            "reason": f"biggest blob {biggest} > {MAX_BIGGEST_BLOB} — likely a stamp/image",
            "biggest_blob": biggest}}

    # ── Size filtering (original algorithm) ──────────────────────────────────
    small_threshold = ((average / CONSTANT_1) * CONSTANT_2) + CONSTANT_3
    big_threshold   = small_threshold * CONSTANT_4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = morphology.remove_small_objects(blobs_labels, small_threshold)

    component_sizes       = np.bincount(cleaned.ravel())
    too_big_mask          = (component_sizes > big_threshold)[cleaned]
    cleaned[too_big_mask] = 0

    if np.count_nonzero(cleaned) == 0:
        return {"has_signature": False, "stats": {"reason": "no blobs survived size filter"}}

    # ── Guard 3: eccentricity ─────────────────────────────────────────────────
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
        return {"has_signature": False, "stats": {"reason": "no qualifying blobs after size filter"}}

    eccentric_ratio = eccentric_count / total_count

    stats = {
        "average_area"    : round(average, 2),
        "biggest_blob"    : biggest,
        "small_threshold" : round(small_threshold, 2),
        "big_threshold"   : round(big_threshold, 2),
        "surviving_pixels": int(np.count_nonzero(cleaned)),
        "eccentric_ratio" : round(eccentric_ratio, 3),
    }

    if debug:
        print(f"        stats -> {stats}")

    if eccentric_ratio < MIN_ECCENTRIC_RATIO:
        return {"has_signature": False, "stats": dict(stats,
            reason=f"eccentric_ratio {eccentric_ratio:.2f} < {MIN_ECCENTRIC_RATIO} — round blobs (stamp/seal), not handwriting")}

    return {"has_signature": True, "stats": stats}


def scan_pdf(pdf_path: str, dpi: int = 150, debug: bool = False):
    """
    Scan every page of a PDF for handwritten signatures.

    Returns
    -------
    pages_with_signature    : list of 1-based page numbers
    pages_without_signature : list of 1-based page numbers
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc         = fitz.open(str(pdf_path))
    total_pages = len(doc)

    print(f"\n  PDF    : {pdf_path.name}")
    print(f"  Pages  : {total_pages}  |  DPI : {dpi}")
    print(f"  {'─'*52}")

    pages_with_signature    = []
    pages_without_signature = []

    for page_num in range(total_pages):
        human_num = page_num + 1
        gray_img  = page_to_image(doc[page_num], dpi=dpi)
        result    = detect_signature(gray_img, debug=debug)

        if result["has_signature"]:
            pages_with_signature.append(human_num)
            status = "SIGNATURE FOUND"
        else:
            pages_without_signature.append(human_num)
            reason = result["stats"].get("reason", "")
            status = f"no signature  [{reason}]" if debug and reason else "no signature"

        print(f"  Page {human_num:>3} / {total_pages}  ->  {status}")

    doc.close()
    return pages_with_signature, pages_without_signature


def main():
    # ── EDIT THESE THREE LINES ───────────────────────────────────────────────
    PDF_PATH = r"C:\Users\User\Documents\internship\high_court\working_code\data\output_part1.pdf"
    DPI      = 150      # raise to 200-300 for faint/small signatures
    DEBUG    = False    # set True to see why each page was accepted/rejected
    # ─────────────────────────────────────────────────────────────────────────

    pages_with_sign, pages_without_sign = scan_pdf(PDF_PATH, dpi=DPI, debug=DEBUG)

    print(f"\n{'='*55}")
    print("  RESULTS  (handwritten signatures only)")
    print(f"{'='*55}")

    print(f"\n  [WITH signature]    ({len(pages_with_sign)}) :")
    if pages_with_sign:
        for p in pages_with_sign:
            print(f"        -> Page {p}")
    else:
        print("        (none)")

    print(f"\n  [WITHOUT signature] ({len(pages_without_sign)}) :")
    if pages_without_sign:
        for p in pages_without_sign:
            print(f"        -> Page {p}")
    else:
        print("        (none)")

    print(f"\n{'='*55}")
    print(f"  Total pages scanned : {len(pages_with_sign) + len(pages_without_sign)}")
    print(f"  Signed              : {len(pages_with_sign)}")
    print(f"  Unsigned            : {len(pages_without_sign)}")
    print(f"{'='*55}\n")

    return pages_with_sign, pages_without_sign


# ── Importable API ────────────────────────────────────────────────────────────
def check_pdf_signatures(pdf_path: str, dpi: int = 150, debug: bool = False):
    """
    Use this when importing as a module.

    Example
    -------
    from signature_detector import check_pdf_signatures
    signed, unsigned = check_pdf_signatures("contract.pdf")
    """
    return scan_pdf(pdf_path, dpi=dpi, debug=debug)


if __name__ == "__main__":
    main()
