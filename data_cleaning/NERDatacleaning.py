import pandas as pd
import requests
from typing import List
import ast
import time
from tqdm import tqdm

def clean_ner_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NER extracted data by removing duplicates and performing substring matching.
    
    Args:
        df: DataFrame with NER columns containing string arrays
    
    Returns:
        DataFrame with cleaned NER data
    """
    def clean_array(arr: str) -> List[str]:
        """Convert string representation of array to actual list and remove duplicates"""
        try:
            if isinstance(arr, str):
                arr = ast.literal_eval(arr)
            return list(set(arr))
        except:
            return []

    def wiki_exists(entity: str) -> bool:
        """Check if entity has a Wikipedia page using the MediaWiki API"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "titles": entity,
                "prop": "info"
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check if the page exists
            pages = data.get("query", {}).get("pages", {})
            return not any(page_id == '-1' for page_id in pages.keys())
        except:
            return False

    def wikify_entity(entity: str) -> str:
        """Check if entity has a Wikipedia page"""
        try:
            if wiki_exists(entity):
                return entity
            time.sleep(0.1)  # Rate limiting to avoid API issues
            return entity
        except:
            return entity

    def remove_substrings(entities: List[str]) -> List[str]:
        """Remove entities that are substrings of other entities"""
        result = entities.copy()
        for i, entity1 in enumerate(entities):
            for entity2 in entities:
                # Skip if same entity or entity1 is longer
                if entity1 == entity2 or len(entity1) >= len(entity2):
                    continue
                # If entity1 is substring of entity2, mark for removal
                if entity1.lower() in entity2.lower():
                    if entity1 in result:
                        result.remove(entity1)
        return result
    
    # NER columns to process
    ner_columns = ['FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 
                  'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART']
    
    # Filter only existing columns
    ner_columns = [col for col in ner_columns if col in df.columns]
    
    # Create a copy of the dataframe
    cleaned_df = df.copy()
    
    # Process each NER column with outer progress bar
    for col in tqdm(ner_columns, desc="Processing columns", position=0):
        print(f"\nProcessing column: {col}")
        
        # Clean arrays and remove duplicates
        cleaned_df[col] = cleaned_df[col].apply(clean_array)
        
        # Get total number of entities for progress bar
        total_entities = sum(len(row[col]) for idx, row in cleaned_df.iterrows() if isinstance(row[col], list))
        
        # Initialize counter for progress
        processed_entities = 0
        
        # Create progress bar for entities within this column
        with tqdm(total=total_entities, desc=f"Processing entities in {col}", position=1, leave=False) as pbar:
            # Process each row
            for idx, row in cleaned_df.iterrows():
                if not isinstance(row[col], list):
                    continue
                
                # Step 1: Wikification
                wikified_entities = []
                for entity in row[col]:
                    wikified = wikify_entity(entity)
                    wikified_entities.append(wikified)
                    processed_entities += 1
                    pbar.update(1)
                
                # Step 2: Substring matching
                final_entities = remove_substrings(wikified_entities)
                
                # Update the dataframe
                cleaned_df.at[idx, col] = final_entities
        
        print(f"Completed processing: {col}")
    
    return cleaned_df

# Example usage:
if __name__ == "__main__":
    input_file = "parsed_data//merged_NER_data_UUID.xlsx"
    output_file = "parsed_data//merged_cleaned_NER_data.xlsx"

    df = pd.read_excel(input_file)
    cleaned_df = clean_ner_data(df)

    cleaned_df.to_excel(output_file, index = False)