import streamlit as st
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
import tempfile
st.set_page_config(page_title="Legal Translator", layout="wide")
st.title("⚖️ English → Bengali Legal Translator")
HF_CACHE = r"D:\HC\hf_cache"
TEMP_ENV = r"D:\HC\temp_env"
@st.cache_resource
def load_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=HF_CACHE, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        cache_dir=HF_CACHE,
        local_files_only=True
    ).to(DEVICE)
    ip = IndicProcessor(inference=True)
    return tokenizer, model, ip, DEVICE
tokenizer, model, ip, DEVICE = load_model()
def load_glossary(file):
    glossary = {}
    if file is None:
        return glossary
    content = file.read().decode("utf-8")
    lines = content.splitlines()
    reader = csv.reader(lines)
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            glossary[row[0].strip()] = row[1].strip()
    return glossary
def clean_page_text(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if re.search(r'SUPREME COURT|S\.C\.R\.|REPORTS|\[\d{4}\]', clean, re.I):
            continue
        if clean in list("ABCDEFGH"):
            continue
        clean = re.sub(r'<.*?>', '', clean)
        filtered.append(clean)
    return "\n".join(filtered)
def split_into_chunks(text, max_tokens=180):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len((current + " " + s).split()) < max_tokens:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks
def clean_output(text):
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def normalize_bengali(text):
    fixes = {
        "রজ্য": "রাজ্য",
        "মমল": "মামলা",
        "ববরণ": "বিবরণ",
        "পশ্চম": "পশ্চিম",
        "ইজর": "ইজারা"
    }
    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)
    return text
LEGAL_LOCKS_POST = {
    "mining lease": "খনি ইজারা",
    "grant order": "মঞ্জুরি আদেশ",
    "respondent": "প্রতিপক্ষ",
    "appellant": "আপিলকারী"
}
def apply_legal_locks_post(text):
    for eng, ben in LEGAL_LOCKS_POST.items():
        text = re.sub(re.escape(eng), ben, text, flags=re.IGNORECASE)
    return text
def apply_glossary(text, glossary):
    for eng, ben in glossary.items():
        text = text.replace(ben, ben)
    return text
uploaded_pdf = st.file_uploader("📄 Upload English PDF", type=["pdf"])
uploaded_glossary = st.file_uploader("📘 Upload Glossary CSV (optional)", type=["csv"])
generate_debug = st.checkbox("Generate Chunk Debug File")
if st.button("🚀 Run Translation"):
    if uploaded_pdf is None:
        st.error("Please upload a PDF first")
    else:
        glossary = load_glossary(uploaded_glossary)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_pdf.read())
            pdf_path = tmp_pdf.name
        pdf = fitz.open(pdf_path)
        doc = Document()
        debug_text = ""
        chunk_counter = 1
        progress = st.progress(0)
        for page_num in range(len(pdf)):
            raw = pdf.load_page(page_num).get_text("text")
            cleaned = clean_page_text(raw)
            chunks = split_into_chunks(cleaned)
            translated_chunks = []
            for chunk in chunks:
                if generate_debug:
                    debug_text += f"ENG_c{chunk_counter}: {chunk}\n"
                batch = ip.preprocess_batch(
                    [chunk],
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
                        num_beams=5,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=3
                    )
                decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                output = ip.postprocess_batch(decoded, lang="ben_Beng")[0]
                output = clean_output(output)
                output = normalize_bengali(output)
                output = apply_legal_locks_post(output)
                output = apply_glossary(output, glossary)
                if generate_debug:
                    debug_text += f"BEN_c{chunk_counter}: {output}\n------\n"
                translated_chunks.append(output)
                chunk_counter += 1
            final = " ".join(translated_chunks)
            if page_num > 0:
                doc.add_page_break()
            p = doc.add_paragraph(final)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.runs[0].font.size = Pt(12)
            progress.progress((page_num + 1) / len(pdf))
            print(f"[PAGE {page_num+1}/{len(pdf)}] Translated")
        docx_path = "output.docx"
        with open(docx_path, "wb") as f:
            doc.save(f)
        st.success("✅ Translation Completed!")
        with open(docx_path, "rb") as f:
            st.download_button("📥 Download DOCX", f, file_name="translated.docx")
        if generate_debug:
            st.download_button(
                "📥 Download Debug TXT",
                debug_text,
                file_name="chunk_debug.txt"
            )