import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Load InLegalBERT
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()
SCRUTINY_CONFIG = {
    "1": {
        "field": "Commercial Arbitration Jurisdiction",
        "query": "mention of commercial division jurisdiction and arbitration petition AP COM number or commercial arbitration case heading",
        "must_have_ents": ["LAW", "ORG"],
        "trigger_words": ["commercial division", "arbitration", "ap(com)", "jurisdiction"],
        "forbidden_patterns": [r"criminal", r"family court"]
    },

    "2": {
        "field": "Prayer under Arbitration Act",
        "query": "prayer clause seeking relief under arbitration and conciliation act sections like section 9 or section 11",
        "must_have_ents": ["LAW"],
        "trigger_words": ["prayer", "section 9", "section 11", "relief", "arbitration act"],
        "forbidden_patterns": [r"valuation", r"property"]
    },

    "3": {
        "field": "Address to Chief Justice",
        "query": "formal addressing to the chief justice or high court authority in petition heading",
        "must_have_ents": ["PERSON", "ORG"],
        "trigger_words": ["chief justice", "honourable", "high court"],
    },

    "4": {
        "field": "Proper Presentation Format",
        "query": "whether petition follows proper filing format including headings index and structured presentation",
        "trigger_words": ["index", "list of dates", "synopsis", "petition"],
    },

    "5": {
        "field": "Client Signature Verification",
        "query": "presence of petitioner or client signature in vakalatnama and after prayer clause",
        "must_have_ents": ["PERSON"],
        "trigger_words": ["signature", "petitioner", "vakalatnama", "signed"],
    },

    "6": {
        "field": "Caveat Intimation",
        "query": "mention of caveat filing or notice to opposite party regarding caveat",
        "trigger_words": ["caveat", "intimation", "notice"],
    },

    "7": {
        "field": "Advocate Enrollment",
        "query": "enrollment number or bar council registration of advocate",
        "must_have_ents": ["CARDINAL"],
        "trigger_words": ["enrollment no", "bar council", "registration"],
    },

    "8": {
        "field": "Advocate Signature on Record",
        "query": "signature of advocate on record in petition document",
        "must_have_ents": ["PERSON"],
        "trigger_words": ["advocate", "signature", "aor", "counsel"],
    },

    "9": {
        "field": "Page Count Compliance",
        "query": "total number of pages in the petition document within prescribed limit like 150 pages",
        "must_have_ents": ["CARDINAL"],
        "trigger_words": ["pages", "page no", "total pages"],
    },

    "10": {
        "field": "Court Number and Section Allocation",
        "query": "mention of court number and section as per determination or listing",
        "must_have_ents": ["CARDINAL"],
        "trigger_words": ["court no", "court number", "section", "listing"],
    },

    "11": {
        "field": "Proper Upload / Filing Check",
        "query": "verification that documents are properly uploaded filed and formatted according to court requirements",
        "trigger_words": ["uploaded", "filed", "annexure", "format", "pdf"],
    }
}

def run_generic_audit():
    with open("nyai_structure_output_1.json", "r") as f:
        data = json.load(f)

    segments = data.get('annotations', [])
    final_report = {}

    for pid, cfg in SCRUTINY_CONFIG.items():
        query_vec = get_embedding(cfg['query'])
        scored_results = []

        for idx, s in enumerate(segments):
            text = s['text']
            text_low = text.lower()
            
            # --- LAYER 1: HARD GATES (Generic) ---
            # 1. Forbidden Patterns: If it looks like a different field, tank the score
            if any(re.search(p, text_low) for p in cfg.get('forbidden_patterns', [])):
                continue 

            # 2. Mandatory Entities: If looking for Property, there MUST be a quantity/number
            s_ents = [e['labels'][0] for e in s.get('entities', [])]
            if not any(ent in s_ents for ent in cfg.get('must_have_ents', [])):
                if not any(tw in text_low for tw in cfg.get('trigger_words', [])):
                    continue

            # --- LAYER 2: SCORING ---
            base_score = float(cosine_similarity(query_vec, get_embedding(text))[0][0])
            
            # Position Boost: Cause Title (2) is usually in first 10% of doc
            pos_weight = 0.2 if pid == "2" and idx < (len(segments) * 0.1) else 0.0
            
            # Trigger Word Boost: (Generic markers like 'Arose' or 'Acres')
            trigger_boost = 0.2 if any(tw in text_low for tw in cfg.get('trigger_words', [])) else 0.0

            total_score = base_score + pos_weight + trigger_boost
            scored_results.append((total_score, text))

        if scored_results:
            best = sorted(scored_results, key=lambda x: x[0], reverse=True)[0]
            final_report[pid] = {
                "field": cfg['field'],
                "extracted": best[1].strip(),
                "confidence": round(best[0], 2)
            }

    return final_report

if __name__ == "__main__":
    report = run_generic_audit()
    print(json.dumps(report, indent=4))