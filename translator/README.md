# Legal Translator: English-to-Bengali Court Document Converter

A specialized machine translation pipeline for converting English court judgments and legal documents into Bengali while preserving legal terminology, formal language structures, and document formatting.

## Overview

This module uses **IndicTrans2**, a state-of-the-art neural machine translation model trained specifically for Indic languages, combined with custom legal domain knowledge, glossary management, and text preprocessing techniques to deliver high-quality Bengali legal translations from English court PDFs.

## Key Features

- **Intelligent Text Processing**: Removes headers, page numbers, and artifacts while preserving legal content
- **Legal Term Preservation**: Pre-translation glossary locking and post-translation legal term standardization
- **Entity Recognition**: Named entity detection for preserving proper nouns (court names, parties, statutes)
- **Batch Translation**: Efficient sentence chunking and batch processing for optimal inference
- **Bengali Normalization**: Corrects common OCR errors and Bengali character inconsistencies
- **Multi-Format Output**: Supports DOCX, PDF, and TXT output formats
- **Streamlit Web UI**: User-friendly interface for PDF uploads and glossary customization

---

## Module Structure

### Core Translation Scripts

#### 1. **app.py** - Streamlit Web Application (Main Interface)
**Purpose**: Interactive web-based translation interface for end users

**Functionality**:
- Uploads English court PDFs through a simple web interface
- Loads the IndicTrans2-1B model with GPU/CPU auto-detection
- Processes multi-page PDFs with progress tracking
- Applies custom glossaries for domain-specific terminology
- Generates DOCX output with proper formatting
- Optional debug chunk export for quality assurance

**Key Processing Pipeline**:
1. Clean page text (remove headers, page numbers, artifacts)
2. Split into translation-friendly chunks (~180 tokens max)
3. Preprocess chunks using IndicProcessor
4. Tokenize and generate translations using the model
5. Postprocess Bengali output
6. Apply glossary and legal term fixes
7. Format output into DOCX with justified text alignment

---

#### 2. **translate_cpu.py** - CPU-Optimized Translation Pipeline
**Purpose**: Production-grade translation pipeline optimized for CPU execution

**Functionality**:
- Uses lighter IndicTrans2 200M model for faster inference on CPU
- Batch processing with configurable batch size (default: 8)
- Paragraph-level grouping with intelligent line merging
- Title detection to preserve document structure
- Generates properly formatted DOCX output

**Key Differences from app.py**:
- Uses 200M model instead of 1B (5x faster, slightly lower quality)
- Optimizes torch threading for multi-core CPU usage
- Groups lines into logical paragraphs before chunking
- Better handling of document structure preservation

---

#### 3. **test.py** - Testing & Debug Pipeline
**Purpose**: Development and quality assurance script

**Functionality**:
- Similar to translate_cpu.py with additional debugging output
- Generates chunk-level debug file tracking English→Bengali mappings
- Validates translation quality at granular level
- Helpful for identifying translation errors and glossary gaps

**Output Files**:
- `chunk_debug.txt` - Line-by-line translation mapping for analysis

---

#### 4. **fixed.py** - Advanced Translation with Term Locking
**Purpose**: Most sophisticated translation with maximum legal term preservation

**Functionality**:
- Pre-translation glossary term locking using special markers (<<<term>>>)
- Entity masking to preserve company names, court names, person names
- Entity unmasking after translation to restore proper nouns
- Comprehensive legal standardization rules
- Safe output cleanup to remove corruption
- Legal Bengali normalization

**Key Enhancements**:
- Locks ~30+ legal terms before translation begins
- Masks named entities using pattern matching (M/s., uppercase titles, proper names)
- Prevents model from mistranslating sensitive legal terminology
- Produces most accurate legal translations

---

#### 5. **extract.py** - Named Entity Recognition (NER)
**Purpose**: Extract legal entities from court documents

**Functionality**:
- Runs OpenNYAI NER pipeline on legal text
- Extracts entities with legal labels:
  - `PERSON`: Judge names, party names
  - `ORG`: Court names, organizations
  - `STATUTE`: Legal acts/statutes
  - `PROVISION`: Legal provisions/sections
  - `COURT`: Court references
- Returns unique entities as JSON for glossary creation or further processing

**Input/Output**:
- Input: Raw text via stdin
- Output: JSON array of entities to stdout

---

#### 6. **to_doc.py** - Document Generator
**Purpose**: Convert translated text to professional document formats

**Functionality**:
- **PDF Generation**: Creates searchable PDFs with Bengali font support
  - Requires TrueType font file (e.g., `Kalpurush.ttf`)
  - Enables complex text shaping for proper Bengali rendering
  - Handles margin markers [ক], [খ], etc.
  - Proper formatting for headers and body text

- **DOCX Generation**: Creates Microsoft Word documents
  - Automatic Unicode Bengali support
  - Maintains text formatting and structure
  - Compatible with all major platforms

**Input**: Structured Bengali text file
**Output**: PDF and/or DOCX documents

---

## Translation Pipeline Details

### Text Cleaning & Preprocessing
```
Raw PDF Text → Remove Headers/Footers → Remove Page Numbers
→ Remove Artifacts (dates, page refs) → Convert to Sentences
→ Split into Chunks (~180-350 tokens depending on model)
```

### Legal Term Preservation Strategies

**1. Pre-translation Glossary Locking** (`fixed.py`):
- Lock terms before translation: "grant order" → "<<<মঞ্জুরি আদেশ>>>"
- Translation doesn't touch locked terms
- Unlock after translation

**2. Entity Masking** (`fixed.py`):
- Mask proper nouns: "M/s. ACME Corp" → "ENT0TOKEN"
- Translate text only
- Restore original entities

**3. Post-translation Term Fixing**:
- Dictionary of common legal terms to fix
- Applied after model output
- Examples:
  - "mining lease" → "খনি ইজারা"
  - "respondent" → "প্রতিপক্ষ"
  - "appellant" → "আপিলকারী"

### Bengali Output Normalization
Corrects OCR and model errors:
- "রজ্য" → "রাজ্য" (state)
- "মমল" → "মামলা" (case)
- "ববরণ" → "বিবরণ" (description)
- "পশ্চম" → "পশ্চিম" (west)
- "ইজর" → "ইজারা" (lease)

---

## Usage

### Option 1: Web Interface (app.py)
```bash
streamlit run app.py
# Opens browser at http://localhost:8501
```
- Upload PDF
- Optionally upload custom glossary CSV
- Click "Run Translation"
- Download translated DOCX

### Option 2: CLI Translation (translate_cpu.py or test.py)
```bash
python translate_cpu.py
# Requires hardcoded PDF path in script
# Outputs: output.docx
```

### Option 3: Advanced Translation (fixed.py)
```bash
python fixed.py
# Uses term locking and entity masking
# Outputs: highest quality translation
```

### Option 4: Entity Extraction (extract.py)
```bash
echo "Your legal text here" | python extract.py
# Outputs: JSON array of extracted entities
```

---

## Glossary Management

### Glossary File Format (`dict.csv`)
```csv
English Term,Bengali Translation
grant order,মঞ্জুরি আদেশ
mining lease,খনি ইজারা
respondent,প্রতিপক্ষ
appellant,আপিলকারী
subject to,শর্তসাপেক্ষে
```

### How Glossary is Applied
1. Load glossary from CSV (sorted by term length, longest first for better matching)
2. Lock terms before translation
3. Apply glossary in post-processing if missed by model
4. Prevents incorrect generic translations of domain-specific terms

---

## Dependencies

### Python Packages
- `streamlit` - Web interface
- `torch` - Neural network inference
- `transformers` - Model loading and inference
- `IndicTransToolkit` - Indic language preprocessing/postprocessing
- `fitz` (PyMuPDF) - PDF reading
- `python-docx` - DOCX document generation
- `fpdf2` - PDF generation with Bengali fonts
- `opennyai` - Named entity recognition

### Model
- **IndicTrans2-1B** (app.py): 1 billion parameters, higher quality, needs GPU
- **IndicTrans2-200M** (translate_cpu.py): 200 million parameters, faster, CPU-viable

### External Resources
- Hugging Face Model Hub: `ai4bharat/indictrans2-en-indic-1B` or `-dist-200M`
- Bengali Font (for PDF): `Kalpurush.ttf` or similar
- HuggingFace cache directory for model weights

---

## Configuration

### Key Parameters (Configurable in Scripts)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MAX_TOKENS` | 180-350 | Max tokens per chunk to prevent OOM |
| `BATCH_SIZE` | 8 | Parallel sentences per batch |
| `num_beams` | 5 (app.py), 1 (cpu) | Beam search width for translation quality |
| `repetition_penalty` | 1.15 | Penalize repeated n-grams |

### Environment Setup
```python
os.environ['HF_HOME'] = r"D:\HC\hf_cache"  # Model cache
os.environ['TEMP'] = r"D:\HC\temp_env"     # Temporary files
```

---

## Output Quality

### Legal Language Preservation
- ✅ Maintains formal structure and tone
- ✅ Preserves cited cases and statutes
- ✅ Protects court abbreviations (HC, SC, etc.)
- ✅ Keeps legal procedural language intact
- ✅ Preserves party designations (Appellant, Respondent)

### Formatting Preservation  
- ✅ Multi-page documents maintained
- ✅ Page breaks inserted appropriately
- ✅ Text alignment (justified)
- ✅ Font sizing and styling
- ✅ Special characters and symbols

---

## Performance Characteristics

| Configuration | Speed | Quality | GPU/CPU |
|--------------|-------|---------|---------|
| **app.py** | ~2-5 sec/page | Highest | GPU |
| **translate_cpu.py** | ~10-15 sec/page | High | CPU |
| **test.py** | ~10-15 sec/page | High | CPU |
| **fixed.py** | ~5-10 sec/page | Highest | GPU/CPU |

---

## Error Handling

### Common Issues & Solutions

**Issue**: "Model not found"
- **Solution**: Ensure HuggingFace cache is configured; run `huggingface-cli download` first

**Issue**: Out of Memory (OOM)
- **Solution**: Reduce `max_tokens` or use smaller 200M model

**Issue**: Garbled Bengali text
- **Solution**: Ensure proper UTF-8 encoding; check Bengali font installation

**Issue**: Incorrect legal terms**
- **Solution**: Create custom glossary CSV file; use `fixed.py` with term locking

---

## Future Enhancements

- [ ] Support for other Indic languages (Hindi, Tamil, etc.)
- [ ] Fine-tuning on legal domain corpora
- [ ] OCR preprocessing for scanned documents
- [ ] Document structure preservation (tables, sections)
- [ ] Batch processing queue for web interface
- [ ] Translation quality scoring/metrics
- [ ] Multi-language glossary support

---

## License & Attribution

- **IndicTrans2 Model**: AI4Bharat (Apache 2.0)
- **OpenNYAI**: For NER capabilities
- **PyMuPDF, python-docx, fpdf2**: Open source libraries

---

## Contact & Support

For issues, improvements, or domain-specific terminologies, refer to the glossary expansion workflow and `extract.py` for entity verification.
