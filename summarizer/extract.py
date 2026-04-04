import json
import os
import sys
from opennyai import Pipeline
from opennyai.utils import Data
from pypdf import PdfReader

def run_opennyai_extraction(pdf_path):
    # 1. Extract text from PDF
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # 2. Create OpenNyAI Data object (Required for preprocessing)
    data = Data([full_text])

    # 3. Initialize Pipeline with CORRECT component names
    # Note: 'NER' and 'Rhetorical_Role' are CASE-SENSITIVE
    use_gpu = True # Set to True if you have a GPU/CUDA configured
    pipeline = Pipeline(components=['NER', 'Rhetorical_Role'], use_gpu=use_gpu, verbose=True)

    # 4. Process the data
    print(f"[*] OpenNyAI: Analyzing structure of {pdf_path}...")
    results = pipeline(data)

    # 5. Save the structured JSON
    # results[0]['annotations'] contains both the entities and the rhetorical roles
    with open("nyai_structure_output_1.json", "w") as f:
        json.dump(results[0], f, indent=4)
        
    print("[+] Structure saved to nyai_structure_output_1.json")

if __name__ == "__main__":
    # Check if a PDF path was passed as an argument
    if len(sys.argv) > 1:
        PDF_FILE = sys.argv[1]
    else:
        # Default fallback
        PDF_FILE = r"D:\HC\output_sample_1.pdf"
        
    if os.path.exists(PDF_FILE):
        run_opennyai_extraction(PDF_FILE)
    else:
        print(f"Error: File not found {PDF_FILE}")