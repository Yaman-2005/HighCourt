import os
import csv
import torch
import re
import fitz
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ---------------------------
# LEGAL STANDARDIZATION
# ---------------------------
LEGAL_LOCKS = {
    r"\bissue for consideration\b": "বিবেচ্য বিষয়",
    r"\bheld\b": "আদালতের সিদ্ধান্ত",
    r"\bit is held\b": "আদালত রায় দিয়েছে যে",
    r"\bit is directed that\b": "নির্দেশ দেওয়া হলো যে",
    r"\bsubject to\b": "শর্তসাপেক্ষে",
    r"\bshall be treated as\b": "বিবেচিত হবে",
    r"\bgrant order\b": "মঞ্জুরি আদেশ",
    r"\bmining lease\b": "খনি ইজারা",
    r"\brespondent\b": "প্রতিপক্ষ",
    r"\bappellant\b": "আপিলকারী"
}

# ---------------------------
# ENV SETUP
# ---------------------------
os.environ['HF_HOME'] = r"D:\HC\hf_cache"
os.environ['TEMP'] = r"D:\HC\temp_env"

# ---------------------------
# LOAD GLOSSARY
# ---------------------------
def load_glossary(csv_path):
    glossary = {}
    if not os.path.exists(csv_path):
        return glossary

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                glossary[row[0].strip()] = row[1].strip()

    return dict(sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True))

# ---------------------------
# TERM LOCKING
# ---------------------------
def lock_terms(text, glossary):
    for eng, ben in glossary.items():
        text = re.sub(
            r'(?<!\w)' + re.escape(eng) + r'(?!\w)',
            f"<<<{ben}>>>",
            text,
            flags=re.IGNORECASE
        )
    return text

def unlock_terms(text):
    return text.replace("<<<", "").replace(">>>", "")

# ---------------------------
# ENTITY DETECTION
# ---------------------------
def extract_entities(text):
    entities = set()

    entities.update(re.findall(r'M/s\.\s*[A-Za-z0-9().,&\- ]+', text))
    entities.update(re.findall(r'\b[A-Z][A-Z\s.&]{3,}\b', text))
    entities.update(re.findall(r'\b(?:[A-Z][a-z]+\s){1,5}[A-Z][a-z]+\b', text))

    return list(entities)

def mask_entities(text, entities):
    entity_map = {}
    for i, ent in enumerate(entities):
        key = f"ENT{i}TOKEN"
        text = re.sub(re.escape(ent), key, text)
        entity_map[key.lower()] = ent
    return text, entity_map

def unmask_entities(text, entity_map):
    for key, val in entity_map.items():
        text = re.sub(key, val, text, flags=re.IGNORECASE)
    return text

# ---------------------------
# CLEAN TEXT
# ---------------------------
def clean_page_text(text):
    lines = text.split('\n')
    filtered = []

    for line in lines:
        clean = line.strip()
        if clean in list("ABCDEFGH"):
            continue
        if re.search(r'SUPREME COURT|S\.C\.R\.|REPORTS|\[\d{4}\]', clean, re.I):
            continue
        if clean:
            filtered.append(clean)

    return " ".join(filtered)

# ---------------------------
# CLAUSE-AWARE CHUNKING
# ---------------------------
def split_into_chunks(text, max_tokens=180):
    sentences = re.split(r'(?<=[.;:])\s+', text)

    chunks = []
    current = ""

    for s in sentences:
        if len((current + s).split()) < max_tokens:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

# ---------------------------
# REPETITION CLEANUP
# ---------------------------
def remove_repetitions(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return text

# ---------------------------
# SAFE CLEANUP
# ---------------------------
def clean_corruption(text):
    text = re.sub(r'[^\u0980-\u09FFA-Za-z0-9\s.,()\-:%]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------
# LEGAL NORMALIZATION
# ---------------------------
def legal_normalize(text):
    replacements = {
        "কোন দ্বিধা নেই": "কোনও অসুবিধা নেই",
        "দেওয়া হয়েছে": "প্রদান করা হয়েছে",
        "বলা হয়েছে": "উল্লেখ করা হয়েছে"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    PDF_PATH = r"D:\HC\translator\eng.pdf"
    GLOSSARY_CSV = "dict.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "ai4bharat/indictrans2-en-indic-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=os.environ['HF_HOME'], local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=os.environ['HF_HOME'],
        local_files_only=True
    ).to(DEVICE)

    ip = IndicProcessor(inference=True)
    glossary = load_glossary(GLOSSARY_CSV)

    pdf = fitz.open(PDF_PATH)
    doc = Document()

    prev_chunk = ""

    for page_num in range(len(pdf)):

        if page_num > 0:
            doc.add_page_break()

        page = pdf.load_page(page_num)
        raw = page.get_text("text")
        print(f"Processing Page {page_num + 1}/{len(pdf)}...")
        # CLEAN
        cleaned = clean_page_text(raw)
        cleaned = cleaned.lower()

        # LOCK TERMS
        cleaned = lock_terms(cleaned, glossary)

        # LEGAL LOCKS
        for pattern, ben in LEGAL_LOCKS.items():
            cleaned = re.sub(pattern, ben, cleaned, flags=re.IGNORECASE)

        # ENTITY MASK
        entities = extract_entities(raw)
        cleaned, entity_map = mask_entities(cleaned, entities)

        # CHUNK
        chunks = split_into_chunks(cleaned)

        translated_chunks = []

        for chunk in chunks:

            input_text = prev_chunk + " " + chunk

            batch = ip.preprocess_batch(
                [input_text],
                src_lang="eng_Latn",
                tgt_lang="ben_Beng"
            )

            inputs = tokenizer(
                batch,
                truncation=True,
                max_length=512,
                padding="longest",
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                tokens = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            output = ip.postprocess_batch(decoded, lang="ben_Beng")[0]

            output = remove_repetitions(output)
            output = clean_corruption(output)
            output = legal_normalize(output)
            output = unlock_terms(output)

            translated_chunks.append(output)

            prev_chunk = chunk

        final = " ".join(translated_chunks)
        final = unmask_entities(final, entity_map)

        p = doc.add_paragraph(final)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.runs[0].font.size = Pt(12)

    doc.save("final_output_1.docx")

    print("✅ DONE — Production-grade translation ready.")