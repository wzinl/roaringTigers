import pandas as pd
import numpy as np
from transformers import pipeline
import logging
from tqdm import tqdm
import torch

def classify_articles(input_file, output_file, labels, batch_size=16):
    """
    Classify articles from Excel with progress tracking
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file
        labels (list): Classification labels
        batch_size (int): Number of articles to process in each batch
    """
    # Load input data
    df = pd.read_excel(input_file)
    
    # Remove empty texts
    df = df[df['Text'].notna() & (df['Text'].str.strip() != '')]
    
    # Initialize classifier
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=device  
    )   
    
    # Prepare results storage
    classification_results = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Classifying Articles"):
        batch = df['Text'].iloc[i:i+batch_size].tolist()
        
        # Classify batch
        batch_results = classifier(
            batch, 
            labels, 
            hypothesis_template="This article is about {}."
        )
        
        # Process batch results
        for result in batch_results:
            classification_results.append({
                'text': result['sequence'],
                'top_label': result['labels'][0],
                'top_score': result['scores'][0],
                **{label: score for label, score in zip(result['labels'], result['scores'])}
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(classification_results)

    score_columns = results_df.columns[2:] 

    results_df['top_score'] = df[score_columns].idxmax(axis=1)
    
    # Save to Excel
    results_df.to_excel(output_file, index=False)
    
    logging.info(f"Classification complete. Results saved to {output_file}")

def main():
    # Configuration
    input_file = r'dataset\news_excerpts_parsed.xlsx'
    output_file = r'dataset\classified_articles.xlsx'
    
    # Comprehensive labels
    labels = [
        # Geopolitical & International Relations
        'UN diplomacy',
        'international conflict',
        'peacekeeping operations',
        'global governance',
        'diplomatic negotiations',
        'international sanctions',
        'geopolitical tensions',
        
        # Humanitarian & Social Issues
        'humanitarian crisis',
        'refugee policy',
        'human rights violations',
        'social justice',
        'humanitarian aid',
        'poverty alleviation',
        'healthcare access',
        'gender equality',
        
        # Environmental & Climate
        'climate change policy',
        'environmental protection',
        'sustainable development',
        'natural disaster response',
        'environmental justice',
        'green technology',
        'conservation efforts',
        
        # Economic & Development
        'economic development',
        'global economic policy',
        'international trade',
        'economic sanctions',
        'infrastructure development',
        'technology transfer',
        'emerging markets',
        
        # Security & Conflict
        'international security',
        'conflict resolution',
        'terrorism',
        'military interventions',
        'arms control',
        'cybersecurity',
        'regional stability',
        
        # Health & Pandemic
        'global health policy',
        'pandemic response',
        'vaccine distribution',
        'healthcare infrastructure',
        'medical research',
        'epidemic management',
        
        # Education & Cultural
        'educational policy',
        'cultural exchange',
        'indigenous rights',
        'academic development',
        'language preservation',
        
        # Technology & Innovation
        'technological policy',
        'digital transformation',
        'innovation ecosystems',
        'tech diplomacy',
        
        # Legal & Governance
        'international law',
        'legal frameworks',
        'governance reform',
        'institutional transparency'
    ]
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Perform classification
    classify_articles(input_file, output_file, labels)

if __name__ == "__main__":
    main()