import streamlit as st
import json
from pypdf import PdfReader
from opennyai import Pipeline
from opennyai.utils import Data
import re
import numpy as np
import subprocess
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yake
def ocr_pdf_cli(file_bytes):
    """Runs OCR on a PDF using pdftoppm and tesseract via subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")

        with open(pdf_path, "wb") as f:
            f.write(file_bytes.read())

        # Convert PDF → images
        subprocess.run([
            "pdftoppm",
            "-png",
            pdf_path,
            os.path.join(temp_dir, "page")
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        text = ""

        for file in sorted(os.listdir(temp_dir)):
            if file.endswith(".png"):
                img_path = os.path.join(temp_dir, file)
                txt_base = img_path.replace(".png", "")

                subprocess.run([
                    "tesseract",
                    img_path,
                    txt_base,
                    "-l", "eng",
                    "--psm", "6"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                with open(txt_base + ".txt", "r", encoding="utf-8", errors="ignore") as f:
                    text += f.read() + "\n"

        return text


# ---------------------------
# OpenNyAI Extraction (MODIFIED to accept text)
# ---------------------------
def run_opennyai_on_text(full_text):
    """Runs OpenNyAI pipeline on raw text instead of PDF."""
    data = Data([full_text])

    pipeline = Pipeline(
        components=['NER', 'Rhetorical_Role'],
        use_gpu=False,
        verbose=False
    )

    results = pipeline(data)
    return results[0]


# ---------------------------
# NORMAL PDF TEXT EXTRACTION
# ---------------------------
def extract_pdf_text(file_bytes):
    """Extracts text from PDF using pypdf (no OCR)."""
    reader = PdfReader(file_bytes)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text


def clean_text(text):
    """Cleans text by removing extra whitespace and references."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return text.strip()


def chunk_long_sentences(text):
    """Splits long sentences at commas to improve readability."""
    return re.sub(r'(?<=,)\s+', '. ', text)


def split_sentences(text):
    """Splits text into sentences using regex (no NLTK)."""
    sentences = re.split(r'\n+|(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


# ---------------------------
# SECTION EXTRACTION
# ---------------------------
def extract_sections(text):
    """Extracts sections based on keywords and their positions in the text."""
    section_keywords = [
        "Issue for Consideration",
        "Held:",
        "FACTUAL MATRIX",
        "THE CHALLENGE",
        "ISSUES INVOLVED"
    ]

    sections = {}
    positions = []

    for keyword in section_keywords:
        idx = text.find(keyword)
        if idx != -1:
            positions.append((idx, keyword))

    positions.sort()

    for i in range(len(positions)):
        start_idx, keyword = positions[i]

        if i + 1 < len(positions):
            end_idx = positions[i + 1][0]
        else:
            end_idx = len(text)

        sections[keyword] = text[start_idx:end_idx]

    if not sections:
        sections["FULL TEXT"] = text

    return sections


# ---------------------------
# SCORING
# ---------------------------
def score_sentence(sentence, section):
    """Assigns a score to a sentence based on keyword presence and section type."""
    score = 0
    s = sentence.lower()

    if section == "Held:":
        if "held" in s: score += 3
        if "void" in s or "invalid" in s: score += 2
        if "set aside" in s or "restored" in s: score += 5
        if "jurisdiction" in s: score += 2
        if "review" in s: score += 2

    if section == "FACTUAL MATRIX":
        if re.search(r'\d{4}', s): score += 3
        if "order" in s: score += 2
        if "petition" in s: score += 2
        if "land" in s: score += 1

    return score


# ---------------------------
# SUMMARIZER
# ---------------------------
def summarize_section(text, section):
    """Summarizes a section by scoring sentences and selecting the top ones."""
    text = clean_text(text)

    if "Headnotes" in text:
        text = text.split("Headnotes")[0]

    text = chunk_long_sentences(text)
    sentences = split_sentences(text)

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    tfidf_scores = np.sum(X.toarray(), axis=1)

    similarity_matrix = cosine_similarity(X)
    np.fill_diagonal(similarity_matrix, 0)

    scores = np.ones(len(sentences))
    for _ in range(20):
        scores = 0.85 * similarity_matrix.dot(scores) + 0.15

    final_scores = []
    for i, sent in enumerate(sentences):
        score = tfidf_scores[i] + scores[i] + score_sentence(sent, section)
        final_scores.append(score)

    if section == "Held:":
        top_n = min(15, len(sentences))
    elif section == "FACTUAL MATRIX":
        top_n = min(12, len(sentences))
    else:
        top_n = min(6, len(sentences))

    ranked = np.argsort(final_scores)[-top_n:]

    return [sentences[i] for i in sorted(ranked)]


# ---------------------------
# KEYWORDS
# ---------------------------
def extract_keywords(text, top_n=10):
    """Extracts keywords using YAKE."""
    kw_extractor = yake.KeywordExtractor(n=3, top=top_n)
    return [kw[0] for kw in kw_extractor.extract_keywords(text)]


# ---------------------------
# OPENNYAI → TEXT
# ---------------------------
def opennyai_to_text(data):
    """Converts OpenNyAI structured output to raw text by concatenating annotations."""
    annotations = data.get('annotations', []) or data.get('data', {}).get('annotations', [])
    return " ".join([ann.get("text", "") for ann in annotations if len(ann.get("text", "")) > 20])


# ---------------------------
# MAIN
# ---------------------------
def summarize_from_text(full_text):
    """Runs the entire summarization pipeline starting from raw text."""
    data = run_opennyai_on_text(full_text)
    raw_text = opennyai_to_text(data)

    sections = extract_sections(raw_text)

    summary = {}
    for section, content in sections.items():
        summary[section] = summarize_section(content, section)

    keywords = extract_keywords(raw_text)

    return summary, keywords, data


# ---------------------------
# PARAGRAPH FORMAT
# ---------------------------
def format_paragraphs(sentences, chunk_size=3):
    """Formats sentences into paragraphs of a specified chunk size."""
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Legal Case Summarizer", layout="wide")

st.title("Legal Case Summarizer (Hybrid NLP + OCR)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded!")

    col1, col2 = st.columns(2)

    # -------- OCR BUTTON --------
    with col1:
        if st.button("🔍 Run OCR"):
            with st.spinner("Running OCR..."):
                uploaded_file.seek(0)
                ocr_text = ocr_pdf_cli(uploaded_file)
                st.session_state["ocr_text"] = ocr_text
                st.success("OCR Completed!")

    # -------- ANALYZE BUTTON --------
    with col2:
        if st.button("⚡ Analyze Document"):
            with st.spinner("Processing..."):

                if "ocr_text" in st.session_state:
                    full_text = st.session_state["ocr_text"]
                else:
                    uploaded_file.seek(0)
                    full_text = extract_pdf_text(uploaded_file)

                summary, keywords, result = summarize_from_text(full_text)

            st.subheader("📊 Structured Summary")

            for section, sentences in summary.items():
                st.markdown(f"### {section}")
                for p in format_paragraphs(sentences, 3):
                    st.write(p)
                    st.write("")
                st.divider()

            st.subheader("🔑 Keywords")
            st.write(", ".join(keywords))

            st.download_button(
                label="Download Raw JSON",
                data=json.dumps(result, indent=4),
                file_name="nyai_output.json",
                mime="application/json"
            )