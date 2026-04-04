import os
import json
import ollama
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. FULL 15-FIELD CONFIGURATION
SCRUTINY_CONFIG = [
    {"id": 1, "field": "Cause Title and Parties", "description": "Full names and residences of Plaintiff and Defendant or Appellants and Respondents.", "focus": "Plaintiff, Defendant, Appellant, Respondent, vs, Versus, Name"},
    {"id": 2, "field": "Territorial Jurisdiction", "description": "Address and Pin Code verification for territorial limits.", "focus": "Address, Pin Code, Jurisdiction, Situated, Kolkata, 700"},
    {"id": 3, "field": "Disability Statement", "description": "Check for statements indicating if any party is a minor or of unsound mind.", "focus": "Minor, Unsound, Disability, Guardian, Age"},
    {"id": 4, "field": "Cause of Action Details", "description": "Facts of when and how the cause of action arose.", "focus": "Cause of Action, Arose, Date, Notice, Order dated"},
    {"id": 5, "field": "Jurisdiction Showing", "description": "Specific facts demonstrating court authority.", "focus": "Jurisdiction, Competence, Authority, Power, Sheriff"},
    {"id": 11, "field": "Suit Valuation and Fees", "description": "Monetary value of the suit and court fees paid.", "focus": "Valuation, Value, Rs., ₹, Fees, Amount, Lakh"},
    {"id": 14, "field": "Immovable Property Description", "description": "Property identification, area (acres), and schedules.", "focus": "Property, Schedule, Boundaries, Area, Acres, homestead, pond"}
    # Add other IDs as needed; IDs 6-10, 12, 13, 15 are omitted here for brevity but follow the same pattern
]

# 2. GLOBAL SETTINGS
MODEL_NAME = 'llama3.2' # Run: ollama pull llama3.2
OUTPUT_LOG = "scrutiny_results.json"
VALID_FIELDS = [p['field'] for p in SCRUTINY_CONFIG]
ALL_KEYWORDS = set([kw.lower() for p in SCRUTINY_CONFIG for kw in p['focus'].split(", ")])

def save_progress(field, data):
    current = {}
    if os.path.exists(OUTPUT_LOG):
        with open(OUTPUT_LOG, "r") as f:
            try: current = json.load(f)
            except: current = {}
    if field not in current: current[field] = []
    current[field].append(data)
    with open(OUTPUT_LOG, "w") as f:
        json.dump(current, f, indent=4)

def process_chunk(chunk):
    content = chunk.page_content
    page_num = chunk.metadata.get("page", "Unknown")
    
    if not any(kw in content.lower() for kw in ALL_KEYWORDS):
        return None

    try:
        # STAGE 1: BROAD SENSITIVITY CHECK (Optimized for Llama 3.2)
        # We look for ANY legal identifiers so we don't miss real_doc_2.pdf
        check_prompt = f"""
        [INST] Analyze the following legal text from Page {page_num}:
        {content[:1000]}
        
        Does this text contain party names (Plaintiff/Defendant), a case title, 
        property details, or a specific court order?
        Answer ONLY with 'YES' or 'NO'. [/INST]
        """
        
        check_resp = ollama.generate(model=MODEL_NAME, prompt=check_prompt, options={"num_predict": 5, "temperature": 0})
        
        if "YES" not in check_resp['response'].upper():
            print(f"[-] Page {page_num}: Skipped (No primary legal data).")
            return None

        print(f"[*] Page {page_num}: Legal data detected. Extracting fields...")

        # STAGE 2: TARGETED EXTRACTION (Strict Zero-Trust for Llama)
        extract_prompt = f"""
        [INST] SYSTEM: You are a Zero-Trust Legal Auditor. Extract data for the fields provided.
        RULES:
        1. Every 'found_value' MUST have a verbatim 'quote'.
        2. If a field is not present, return "NOT_FOUND" for that field.
        3. Do NOT confuse case citations (SCC, SCR) with 'Suit Valuation'.
        4. Do NOT confuse procedural hearing notices with 'Disability Statements'.

        FIELDS TO EXTRACT:
        {json.dumps([{f['field']: f['description']} for f in SCRUTINY_CONFIG])}

        TEXT:
        {content}

        RETURN ONLY VALID JSON:
        {{ "Field Name": {{ "found_value": "data", "confidence_score": 10, "quote": "verbatim text" }} }} [/INST]
        """
        
        final_resp = ollama.generate(model=MODEL_NAME, prompt=extract_prompt, format='json', options={"temperature": 0, "num_ctx": 4096})
        batch_res = json.loads(final_resp['response'])
        
        extracted_data = {}
        for field_name, data in batch_res.items():
            # Match the AI's key back to our valid config fields and filter low confidence
            if field_name in VALID_FIELDS and data.get('found_value') != "NOT_FOUND":
                if data.get('confidence_score', 0) >= 7:
                    data['page_number'] = page_num
                    save_progress(field_name, data)
                    extracted_data[field_name] = data
                    print(f"    [+] Found: {field_name}")
        
        return extracted_data if extracted_data else None
    except Exception as e:
        print(f"    [!] Error on Page {page_num}: {e}")
        return None

def run_scrutiny_turbo(pdf_path):
    print(f"[*] Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    docs = [Document(page_content=p.extract_text(), metadata={"page": i+1}) for i, p in enumerate(reader.pages)]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    chunks = text_splitter.split_documents(docs)
    
    print(f"[*] Analyzing {len(chunks)} blocks with 2 Workers...")

    final_results = {p['field']: [] for p in SCRUTINY_CONFIG}
    with ThreadPoolExecutor(max_workers=2) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))

    for res in chunk_results:
        if res:
            for field, data in res.items():
                if field in final_results:
                    final_results[field].append(data)
    
    return final_results

if __name__ == "__main__":
    PDF_FILE = "real_doc.pdf" # Works for real_doc.pdf and sample_calcutta... as well
    if os.path.exists(PDF_FILE):
        report = run_scrutiny_turbo(PDF_FILE)
        print("\n" + "="*60 + "\nFINAL SCRUTINY REPORT\n" + "="*60)
        for field, findings in report.items():
            print(f"\nFIELD: {field}")
            if not findings:
                print("  > NOT FOUND")
            else:
                for f in sorted(findings, key=lambda x: x.get('confidence_score', 0), reverse=True)[:1]:
                    print(f"  [{f['confidence_score']}/10] {f['found_value']} (Page {f['page_number']})")
    else:
        print(f"[!] File {PDF_FILE} not found.")