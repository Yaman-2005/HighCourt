import json
import re
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import yake


# -----------------------------
# 1. LOAD JSON
# -----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]["text"]


# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return text.strip()


# -----------------------------
# 3. SPLIT SENTENCES (NO NLTK)
# -----------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


# -----------------------------
# 4. SECTION EXTRACTION
# -----------------------------
def extract_sections(text):
    section_keywords = [
        "Issue for Consideration",
        "Held:",
        "FACTUAL MATRIX",
        "THE CHALLENGE",
        "ISSUES INVOLVED"
    ]

    sections = defaultdict(str)

    for keyword in section_keywords:
        if keyword in text:
            parts = text.split(keyword, 1)[1]
            sections[keyword] = parts[:2000]

    return sections


# -----------------------------
# 5. TF-IDF SUMMARY
# -----------------------------
def tfidf_summary(sentences, top_n=3):
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    scores = np.sum(X.toarray(), axis=1)
    ranked = np.argsort(scores)[-top_n:]

    return [sentences[i] for i in sorted(ranked)]


# -----------------------------
# 6. TEXTRANK (PURE IMPLEMENTATION)
# -----------------------------
def textrank_summary(sentences, top_n=3):
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(X)

    # Remove self-links
    np.fill_diagonal(similarity_matrix, 0)

    scores = np.ones(len(sentences))

    # PageRank iterations
    for _ in range(20):
        scores = 0.85 * similarity_matrix.dot(scores) + 0.15

    ranked = np.argsort(scores)[-top_n:]

    return [sentences[i] for i in sorted(ranked)]


# -----------------------------
# 7. KEYWORDS (YAKE)
# -----------------------------
def extract_keywords(text, top_n=10):
    kw_extractor = yake.KeywordExtractor(n=2, top=top_n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]


# -----------------------------
# 8. HYBRID SECTION SUMMARY
# -----------------------------
def summarize_section(text):
    text = clean_text(text)
    sentences = split_sentences(text)

    tfidf_part = tfidf_summary(sentences, 3)
    textrank_part = textrank_summary(sentences, 3)

    combined = list(dict.fromkeys(tfidf_part + textrank_part))
    return combined[:5]


# -----------------------------
# 9. MAIN PIPELINE
# -----------------------------
def summarize_json(path):
    raw_text = load_json(path)
    sections = extract_sections(raw_text)

    final_summary = {}

    for section, content in sections.items():
        final_summary[section] = summarize_section(content)

    keywords = extract_keywords(raw_text)

    return final_summary, keywords


# -----------------------------
# 10. OUTPUT
# -----------------------------
def print_summary(summary, keywords):
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for section, sentences in summary.items():
        print(f"\n🔹 {section.upper()}")
        for s in sentences:
            print(f"- {s}")

    print("\n" + "="*60)
    print("KEYWORDS")
    print("="*60)
    print(", ".join(keywords))


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    path = "nyai_structure.json"

    summary, keywords = summarize_json(path)
    print_summary(summary, keywords)