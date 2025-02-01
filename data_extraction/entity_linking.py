import pandas as pd
import spacy
from spacy.language import Language
from transformers import pipeline
from spacy.tokens import Doc, Span
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set extensions
Doc.set_extension("relations", default=[], force=True)
Span.set_extension("linkedEntities", default=[], force=True)

@Language.factory("rebel")
def create_rebel_component(nlp, name, device=0, model_name="Babelscape/rebel-large"):
    relation_extractor = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
        batch_size=32  # Increased batch size for better GPU utilization
    )
    return RebelComponent(relation_extractor)

class RebelComponent:
    def __init__(self, relation_extractor):
        self.relation_extractor = relation_extractor
    
    def __call__(self, doc):
        if len(doc.text) > 1000:
            doc._.relations = []
            return doc
            
        try:
            results = self.relation_extractor(doc.text, max_length=128)  # Add max_length to speed up generation
            doc._.relations = results
        except Exception as e:
            print(f"Error in relation extraction: {e}")
            doc._.relations = []
        return doc

def process_batch(batch_data, nlp):
    """Process a single batch of data"""
    results = []
    for text in batch_data:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            results.append([])
            continue
            
        # Truncate long texts
        if len(text) > 500:
            text = text[:500]
            
        try:
            # Use disable to skip unnecessary pipeline components
            doc = nlp(text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            links = []
            
            if doc.ents:
                for ent in doc.ents:
                    if hasattr(ent._, 'linkedEntities') and ent._.linkedEntities:
                        for link in ent._.linkedEntities:
                            links.append({
                                "entity": ent.text,
                                "link_id": link.get_id(),
                                "description": link.get_description()
                            })
            results.append(links)
        except Exception as e:
            print(f"Error processing entity: {text[:50]}..., Error: {e}")
            results.append([])
            
    return results

def process_dataframe(df, nlp, batch_size=32):
    """Process the dataframe using parallel processing"""
    headers = list(df.columns[1:])
    max_workers = min(len(headers), 4)  # Limit number of parallel workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key in headers:
            print(f"Processing column: {key}")
            futures = []
            all_linked_entities = []
            
            # Split data into batches
            for i in range(0, len(df), batch_size):
                batch = df[key].iloc[i:i+batch_size].tolist()
                futures.append(
                    executor.submit(process_batch, batch, nlp)
                )
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc=f"Processing {key}") as pbar:
                for future in as_completed(futures):
                    all_linked_entities.extend(future.result())
                    pbar.update(1)
            
            # Assign results back to dataframe
            df[f'{key}_linked_entities'] = all_linked_entities
            
    return df

def main():
    # Disable unnecessary spaCy pipeline components
    nlp = spacy.load("en_core_web_trf", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
    # Configure GPU/CPU device and optimize for inference
    if torch.cuda.is_available():
        device = 0
        torch.cuda.empty_cache()  # Clear GPU memory
        # Set torch to inference mode for better performance
        torch.set_grad_enabled(False)
    else:
        device = -1
    
    # Add the rebel component with optimized settings
    if "rebel" not in nlp.pipe_names:
        nlp.add_pipe("rebel", after="ner", config={
            'device': device,
            'model_name': 'Babelscape/rebel-large'
        })
    
    try:
        # Read data in chunks for better memory management
        df = pd.read_excel('extracted_info.xlsx', engine='openpyxl', nrows=10) # Testing only 10 rows
        
        # Process the dataframe with optimized batch size
        df = process_dataframe(df, nlp, batch_size=32)
        
        # Save results
        df.to_excel('linked_entities.xlsx', index=False, engine='openpyxl')
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()