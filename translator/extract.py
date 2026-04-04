import json, sys
from opennyai import Pipeline
from opennyai.utils import Data

def run_extraction(text):
    # Only need NER for protection discovery
    pipeline = Pipeline(components=['NER'], use_gpu=True, verbose=False)
    data = Data([text])
    results = pipeline(data)
    
    # Extract unique legal entities
    dynamic_set = set()
    for ent in results[0].get('annotations', []):
        if any(lbl in ent.get('labels', []) for lbl in ['PERSON', 'ORG', 'STATUTE', 'PROVISION', 'COURT']):
            text = ent.get('text', '').strip()
            if len(text) > 3: dynamic_set.add(text)
    
    return list(dynamic_set)

if __name__ == "__main__":
    # Receive text from the main script via stdin
    raw_text = sys.stdin.read()
    entities = run_extraction(raw_text)
    print(json.dumps(entities))