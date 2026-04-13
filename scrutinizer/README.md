# Scrutinizer: Legal Document Audit & Signature Detection

A comprehensive system for automated legal document scrutiny and verification, combining large language models (LLM-based field extraction), deep learning embeddings, and computer vision for signature detection in court documents and legal petitions.

## Overview

Scrutinizer performs two complementary functions:

1. **Document Audit/Scrutiny**: Extracts critical legal fields from complex court documents using:
   - **LLM Approach** (batch.py): Llama 3.2 with zero-trust auditing and quoted evidence verification
   - **Embedding Approach** (get_data.py): InLegalBERT semantic similarity with 11-field legal petition validation

2. **Signature Detection & Verification**: Identifies handwritten signatures in legal PDFs while distinguishing them from:
   - Rubber stamps and seals (circular signatures)
   - Logos and printed images
   - QR codes and barcodes
   - Text blocks and tables

---

## Module Structure

### Part A: Document Audit & Scrutiny

#### 1. **batch.py** - LLM-Based Scrutiny (Llama 3.2)
**Purpose**: Extract 15 critical legal fields from documents using Llama 3.2 with zero-trust auditing

**Key Features**:
- **Two-Stage LLM Pipeline**:
  - Stage 1: Broad sensitivity check (detects if page contains legal data)
  - Stage 2: Targeted extraction with quoted evidence (zero-trust auditing)
- **15 Legal Fields** (configurable):
  1. Cause Title and Parties
  2. Territorial Jurisdiction
  3. Disability Statement
  4. Cause of Action Details
  5. Jurisdiction Showing
  6. Suit Valuation and Fees
  7. Immovable Property Description
  (+ 8 additional fields for complete legal document audit)

- **Confidence Scoring**: Only reports fields with confidence ≥ 7/10
- **Verbatim Quotes**: Every extracted value includes original text quote
- **Multi-worker Processing**: ThreadPoolExecutor for parallel chunk analysis
- **Keyword Filtering**: Smart ALL_KEYWORDS detection to avoid processing junk pages

**Processing Pipeline**:
```
PDF → Extract Text → Split into Chunks (2500 chars)
→ Keyword Filter → LLM Sensitivity Check 
→ LLM Targeted Extraction → Confidence Filter (≥7)
→ Quote Validation → JSON Output
```

**Configuration**:
```python
SCRUTINY_CONFIG = [
    {
        "field": "Cause Title and Parties",
        "description": "Full names and residences of Plaintiff and Defendant",
        "focus": "Plaintiff, Defendant, Appellant, Respondent, vs, Versus, Name"
    },
    # ... more fields
]
```

**Output Format**:
```json
{
  "Cause Title and Parties": [
    {
      "found_value": "Plaintiff Name vs Defendant Name",
      "confidence_score": 9,
      "quote": "In the matter of [full verbatim quote]",
      "page_number": 1
    }
  ],
  "Territorial Jurisdiction": [...]
}
```

**Requirements**:
- Ollama with Llama 3.2 model installed (`ollama pull llama3.2`)
- PyPDF for PDF text extraction
- LangChain for text splitting

---

#### 2. **get_data.py** - InLegalBERT Embeddings-Based Audit
**Purpose**: Extract 11 specialized legal petition fields using semantic similarity with InLegalBERT

**Key Features**:
- **InLegalBERT Embeddings**: Specialized legal language model for semantic understanding
- **11 Audit Fields**:
  1. Commercial Arbitration Jurisdiction
  2. Prayer under Arbitration Act
  3. Address to Chief Justice
  4. Proper Presentation Format
  5. Client Signature Verification
  6. Caveat Intimation
  7. Advocate Enrollment
  8. Advocate Signature on Record
  9. Page Count Compliance
  10. Court Number and Section Allocation
  11. Proper Upload / Filing Check

- **Three-Layer Validation**:
  - Layer 1: Hard gates (forbidden patterns, mandatory entities)
  - Layer 2: Semantic scoring (cosine similarity + position boost + trigger words)
  - Layer 3: Confidence thresholding

**Scoring Algorithm**:
```
base_score = cosine_similarity(query_embedding, text_embedding)
position_boost = 0.2 if (field is near doc start AND position matters)
trigger_boost = 0.2 if (legal keywords detected)
final_score = base_score + position_boost + trigger_boost
```

**Configuration Example**:
```python
{
    "field": "Commercial Arbitration Jurisdiction",
    "query": "mention of commercial division jurisdiction and arbitration petition",
    "must_have_ents": ["LAW", "ORG"],  # Required NER entities
    "trigger_words": ["commercial division", "arbitration", "ap(com)"],
    "forbidden_patterns": [r"criminal", r"family court"]  # Field disqualifiers
}
```

**Output Format**:
```json
{
  "1": {
    "field": "Commercial Arbitration Jurisdiction",
    "extracted": "Commercial division jurisdiction - Petition CP(COM) 1234/2024",
    "confidence": 0.87
  }
}
```

**Requirements**:
- InLegalBERT (`law-ai/InLegalBERT`)
- PyTorch for embeddings
- OpenNyAI output JSON

---

### Part B: Signature Detection & Verification

#### 3. **signature_detector.py** - Core Signature Detection
**Purpose**: Detect handwritten ink signatures in PDF pages using computer vision

**Key Features**:
- **Connected Component Analysis**: Binary thresholding + blob detection
- **Eccentricity-Based Filtering**: Distinguishes elongated pen strokes from round stamps
- **3-Guard System**:
  - Guard 1: Reject pages with high average blob area (dense print)
  - Guard 2: Reject pages with very large single blobs (images/stamps)
  - Guard 3: Reject pages with mostly circular blobs (rubber stamps)

**Detection Algorithm**:

1. **Binarization**:
```python
binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
blobs = measure.label(binary > binary.mean())
```

2. **Size Filtering**:
```python
average_area = total_area / blob_count
small_threshold = ((average / 84) * 250) + 100
big_threshold = small_threshold * 18
# Keeps blobs between thresholds
```

3. **Eccentricity Analysis**:
```python
eccentric_ratio = elongated_blobs / total_blobs
# If eccentric_ratio < 0.4 → likely stamp (too round)
# If eccentric_ratio ≥ 0.4 → likely signature (elongated strokes)
```

**Detection Constants** (tunable):
```python
CONSTANT_1 = 84           # Average scaling factor
CONSTANT_2 = 250          # Small threshold multiplier
CONSTANT_3 = 100          # Small threshold offset
CONSTANT_4 = 18           # Big threshold multiplier

MAX_AVERAGE_AREA = 100    # Reject if avg area > 100px (dense print)
MAX_BIGGEST_BLOB = 5000   # Reject if largest blob > 5000px (stamp/image)
MIN_ECCENTRICITY = 0.5    # A blob is "stroke-like" if eccentricity > 0.5
MIN_ECCENTRIC_RATIO = 0.4 # At least 40% of blobs must be stroke-like
```

**Output Format**:
```python
{
  "pages_with_signature": [1, 5, 6, 10, 11],      # 1-based page numbers
  "pages_without_signature": [2, 3, 4, 7, 8, 9],
  "detailed_stats": {
    "average_area": 45.2,
    "biggest_blob": 3200,
    "eccentric_ratio": 0.65,
    "surviving_pixels": 12450
  }
}
```

**Usage**:
```python
pages_with_sig, pages_without_sig = scan_pdf("document.pdf", dpi=150, debug=False)
```

---

#### 4. **signature_debug.py** - Parameter Tuning & Visualization
**Purpose**: Debug and visualize signature detection at the blob level

**Features**:
- **Color-Coded Debug Images**:
  - 🟢 GREEN: KEPT blobs (algorithm thinks these are signatures)
  - 🔴 RED: REMOVED blobs (too big - text/table/border)
  - ⚪ LIGHT GREY: REMOVED blobs (too small - noise/dust)

- **Per-Page Metrics**:
  - Average blob area
  - Biggest blob size
  - Small/big thresholds
  - Surviving pixel count
  - Eccentricity ratio

- **Guard Alerts**: Indicates which guard(s) reject the page

**Output**: `debug_output/page_XXX_debug.png` for each processed page with colored overlay and legend

**Usage**:
```bash
python signature_debug.py
# Edit PDF_PATH and PAGE_NUMBERS at top
# Outputs: debug_output/page_001_debug.png, page_002_debug.png, etc.
```

**Visualization Example**:
```
Page 5:
  average blob area   : 42.5 px
  biggest blob        : 3100 px
  small_threshold     : 107.2  (blobs BELOW = noise, removed)
  big_threshold       : 1929.6 (blobs ABOVE = text/table, removed)
  => kept blobs are between 107 and 1930 px
  surviving pixels    : 8234
  eccentric_ratio     : 0.72  (need >= 0.4 to be 'handwriting')
  Verdict             : SIGNATURE FOUND
  Debug image saved   -> debug_output/sample_1/page_005_debug.png
```

---

#### 5. **signature_accuracy.py** - Model Evaluation
**Purpose**: Evaluate signature detection accuracy against ground truth

**Metrics Computed**:
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity**: TN / (TN + FP)

**Output Format**:
```
PDF            : output_sample_2.pdf
Total pages    : 36
Evaluating     : pages 1 – 36
GT signed pages: [5, 6, 10, 11, 12, 29, 30]
────────────────────────────────────────────────────────────────
Page      GT        Predicted        Result              Reason
────────────────────────────────────────────────────────────────
  1     EMPTY       NO_SIG          CORRECT_NEGATIVE    blank page
  5     SIGNED      SIGNED          TRUE_POSITIVE       eccentric_ratio 0.68
  6     SIGNED      SIGNED          TRUE_POSITIVE       eccentric_ratio 0.71
  ...
────────────────────────────────────────────────────────────────
FINAL METRICS:
  Accuracy   : 0.92
  Precision  : 0.95
  Recall     : 0.89
  F1         : 0.92
  Specificity: 0.93
```

**Editable Ground Truth**:
```python
GROUND_TRUTH_SIGNED = {
    5, 6, 10, 11, 12, 29, 30  # Set of 1-based page numbers with signatures
}
```

**Usage**:
```bash
python signature_accuracy.py
# Edit PDF_PATH and GROUND_TRUTH_SIGNED at top
# Outputs: Console report with metrics
```

---

#### 6. **convert.py** - Format Conversion Utility
**Purpose**: Convert OpenNyAI structured JSON to PreSumm format

**Conversion Process**:
```
OpenNyAI JSON (annotations with text)
→ Extract and Clean Sentences
→ Tokenize (remove special chars, lowercase split)
→ Create PreSumm Object (src, tgt, cpd)
→ Output JSON
```

**Input Format** (OpenNyAI):
```json
{
  "id": "doc_123",
  "annotations": [
    { "text": "First sentence in document." },
    { "text": "Second sentence here." }
  ]
}
```

**Output Format** (PreSumm):
```json
[
  {
    "src": [
      ["first", "sentence", "in", "document"],
      ["second", "sentence", "here"]
    ],
    "tgt": [],
    "cpd": "doc_123"
  }
]
```

**Usage**:
```bash
python convert.py
# Reads: nyai_structure.json
# Outputs: to_summarize.json
```

---

## Complete Workflow

### Scenario 1: Audit a Court Petition
```bash
# Step 1: Extract text and structure with OpenNyAI
python ../summarizer/extract.py petition.pdf
# Output: nyai_structure_output_1.json

# Step 2: Run batch scrutiny with Llama
python batch.py
# Output: scrutiny_results.json (15 fields)

# Step 3: Run embedding-based audit
python get_data.py
# Output: audit_report.json (11 petition fields)

# Step 4: Detect signatures
python signature_detector.py
# Output: Console report with signed pages
```

### Scenario 2: Debug Signature Detection
```bash
# Step 1: Run debug visualization
python signature_debug.py
# Output: debug_output/page_001_debug.png, page_002_debug.png, ...

# Step 2: Adjust constants in signature_detector.py based on visualization

# Step 3: Evaluate with ground truth
python signature_accuracy.py
# Output: Accuracy metrics
```

---

## Dependencies

### Python Libraries
- **Document Processing**:
  - `pypdf` - PDF text extraction
  - `fitz` (PyMuPDF) - PDF rendering to images
  
- **LLM & Embeddings**:
  - `ollama` - Llama 3.2 inference
  - `transformers` - InLegalBERT model loading
  - `torch` - Tensor operations
  
- **Text Processing**:
  - `langchain_text_splitters` - RecursiveCharacterTextSplitter
  - `langchain_core` - Document class
  
- **Computer Vision**:
  - `opencv-python` - Image processing
  - `scikit-image` - Connected components, morphology
  - `numpy` - Array operations
  
- **NLP & Metrics**:
  - `scikit-learn` - Cosine similarity
  - `yake` (optional) - Keyword extraction

### System & Model Requirements

**For Batch Scrutiny**:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.2    # ~2GB model
# Ollama runs on port 11434
```

**For Embeddings Audit**:
```bash
# InLegalBERT downloads automatically on first use
# Requires: 4GB+ RAM, GPU optional
```

**Installation**:
```bash
pip install pypdf fitz transformers torch opencv-python scikit-image numpy scikit-learn langchain-text-splitters langchain-core ollama
```

---

## Configuration & Tuning

### Batch Scrutiny Configuration
```python
SCRUTINY_CONFIG = [
    {
        "id": 1,
        "field": "Cause Title and Parties",
        "description": "Full names and residences...",
        "focus": "Plaintiff, Defendant, Appellant, ..."  # Keywords
    }
]

MODEL_NAME = 'llama3.2'
VALID_FIELDS = [p['field'] for p in SCRUTINY_CONFIG]
```

### Signature Detection Constants
Edit in both `signature_detector.py` and `signature_debug.py`:

```python
CONSTANT_1 = 84           # Increase → stricter size filtering
CONSTANT_2 = 250          # Increase → bigger threshold range
CONSTANT_3 = 100          # Increase → minimum size floor
CONSTANT_4 = 18           # Increase → stricter big threshold

MAX_AVERAGE_AREA = 100    # Decrease → rejects dense pages earlier
MAX_BIGGEST_BLOB = 5000   # Decrease → rejects large images earlier
MIN_ECCENTRICITY = 0.5    # Increase → rejects rounder blobs
MIN_ECCENTRIC_RATIO = 0.4 # Increase → requires more stroke-like blobs
```

### DPI Adjustment
```python
DPI = 150   # Default; increase to 200-300 for faint/small signatures
```

---

## Performance Characteristics

| Operation | Time | Hardware |
|-----------|------|----------|
| Batch Scrutiny (1 page) | 2-5 sec | CPU (~2GB RAM) |
| Batch Scrutiny (10 pages) | 20-50 sec | CPU with 2 workers |
| Embedding Audit (1 doc) | 1-3 sec | CPU/GPU |
| Signature Detection (100 pages) | 10-30 sec | CPU |
| Debug Visualization (10 pages) | 5-15 sec | CPU |

---

## Output Examples

### Batch Scrutiny Output
```json
{
  "Cause Title and Parties": [
    {
      "found_value": "State of West Bengal vs Jai Hind Private Limited",
      "confidence_score": 9,
      "quote": "[2026] 2 S.C.R. 497 : State of West Bengal & Ors. v. Jai Hind Pvt. Ltd.",
      "page_number": 1
    }
  ],
  "Territorial Jurisdiction": [
    {
      "found_value": "Kolkata, 700001",
      "confidence_score": 8,
      "quote": "Writers Buildings, Kolkata - 700001",
      "page_number": 5
    }
  ]
}
```

### Embedding Audit Output
```json
{
  "1": {
    "field": "Commercial Arbitration Jurisdiction",
    "extracted": "Commercial Division - AP(COM) petition filed",
    "confidence": 0.87
  },
  "5": {
    "field": "Client Signature Verification",
    "extracted": "Petitioner signature affixed on page 12",
    "confidence": 0.92
  }
}
```

### Signature Detection Output
```
PDF    : sample_2.pdf
Pages  : 36  |  DPI : 150
────────────────────────────────
Page   1 / 36  ->  no signature
Page   5 / 36  ->  SIGNATURE FOUND
Page   6 / 36  ->  SIGNATURE FOUND
Page  10 / 36  ->  SIGNATURE FOUND
...
Page  36 / 36  ->  no signature
```

---

## Troubleshooting

**Issue**: "ollama: command not found"
- **Solution**: Install Ollama from https://ollama.ai/ and ensure it's in PATH

**Issue**: "ModuleNotFoundError: transformers"
- **Solution**: `pip install transformers torch`

**Issue**: Signature detection all pages as "no signature"
- **Solution**: Increase DPI (200-300) or decrease MIN_ECCENTRIC_RATIO

**Issue**: Too many false positives (stamps detected as signatures)
- **Solution**: Increase CONSTANT_1 or MAX_AVERAGE_AREA

**Issue**: Embedding audit returns all "NOT_FOUND"
- **Solution**: Ensure document contains required trigger keywords for fields

---

## Legal Document Compliance

### Fields Extracted (Batch Scrutiny)
✅ Parties and Cause Title  
✅ Territorial/Subject Matter Jurisdiction  
✅ Disability/Minority Statements  
✅ Cause of Action  
✅ Jurisdiction Showing  
✅ Suit Valuation & Fees  
✅ Property Descriptions  

### Petition Fields Verified (Embedding Audit)
✅ Commercial Arbitration Jurisdiction  
✅ Prayer Clauses  
✅ Chief Justice Address  
✅ Format Compliance  
✅ Client Signatures  
✅ Advocate Credentials  
✅ Page Count Compliance  
✅ Filing Status  

### Signature Verification
✅ Handwritten Ink Detection  
✅ Stamp/Seal Rejection  
✅ Printed Image Rejection  
✅ QR Code/Barcode Rejection  
✅ Accuracy Metrics  

---

## Enhancement Opportunities

- [ ] Multi-page aggregation (combine findings across pages)
- [ ] Confidence score evolution tracking
- [ ] Batch export to CSV/Excel
- [ ] Web UI for petition verification
- [ ] Integration with court management systems (CMS)
- [ ] Real-time signature zone mapping
- [ ] Multi-language legal document support
- [ ] OCR preprocessing for scanned documents
- [ ] Custom field definitions per jurisdiction
- [ ] API endpoint deployment

---

## License & Attribution

- **Llama 3.2**: Meta AI (Apache 2.0)
- **InLegalBERT**: Law AI / OpenCourts (Fine-tuned BERT for legal domain)
- **PyPDF, PyMuPDF, OpenCV**: Various open-source licenses
- **scikit-image**: BSD license

---

## References

### Computer Vision (Signature Detection)
- Connected Component Analysis (Image Labeling)
- Blob Detection via Binary Thresholding
- Eccentricity-based shape analysis
- Morphological Filtering (remove_small_objects)

### NLP (Document Audit)
- Zero-Trust Auditing with LLM evidence quotes
- Semantic Similarity via Transformer Embeddings
- Position-based scoring for document structure
- Multi-layer validation gates

### Legal Reference
- West Bengal Estates Acquisition Act, 1953
- Commercial Division Petition Guidelines
- High Court Filing Standards
- Advocate-on-Record Requirements

---

## Support & Feedback

For document-specific issues, parameter tuning guidance, or field definition expansion, refer to:
- `signature_debug.py` for visualization-based tuning
- `signature_accuracy.py` for metric-based evaluation
- SCRUTINY_CONFIG in scripts for field customization
