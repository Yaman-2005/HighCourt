import json
import re

def convert_to_presumm_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # OpenNyAI structure puts segments in data['annotations'] or just 'annotations'
    annotations = data.get('annotations', [])
    if not annotations and 'data' in data:
        annotations = data['data'].get('annotations', [])

    # 1. Extract and Clean Sentences
    # The summarizer needs a list of tokenized sentences
    src_tokens = []
    for ann in annotations:
        text = ann.get('text', '').strip()
        if not text:
            continue
        
        # Simple tokenization: remove special characters and split by whitespace
        # You can use NLTK or SpaCy here for better accuracy
        clean_text = re.sub(r'[^\w\s]', '', text)
        tokens = clean_text.lower().split()
        
        if tokens:
            src_tokens.append(tokens)

    # 2. Construct the PreSumm Object
    # 'tgt' is for the gold summary (empty during inference)
    # 'cpd' is for the document ID or metadata
    presumm_data = [{
        "src": src_tokens,
        "tgt": [],
        "cpd": data.get('id', 'doc_1')
    }]

    # 3. Save as the expected format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(presumm_data, f)

    print(f"[+] Conversion complete. Saved to: {output_path}")
    print(f"[i] Processed {len(src_tokens)} sentences.")

if __name__ == "__main__":
    convert_to_presumm_format("nyai_structure.json", "to_summarize.json")