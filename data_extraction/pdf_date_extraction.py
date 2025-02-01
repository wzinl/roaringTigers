import pdfplumber
import re
import os
from datetime import datetime
import pandas as pd

def extract_date_from_text(text):
    """
    Extract date from text using various common date formats.
    Returns the first found date or None if no date is found.
    """
    # Common date patterns (add more patterns as needed)
    date_patterns = [
        # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        # Month DD, YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        # DD Month YYYY
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        # YYYY-MM-DD
        r'\b\d{4}-\d{2}-\d{2}\b',
        # Abbreviated months
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    return None

def process_pdf_folder(folder_path):
    """
    Process all PDF files in the specified folder and extract dates.
    Returns a DataFrame with filename and extracted date.
    """
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                with pdfplumber.open(file_path) as pdf:
                    # Extract text from first page only (modify as needed)
                    first_page = pdf.pages[0]
                    text = first_page.extract_text()
                    
                    # Try to find a date in the text
                    date_found = extract_date_from_text(text)
                    
                    results.append({
                        'filename': filename,
                        'extracted_date': date_found,
                        'status': 'success' if date_found else 'no date found'
                    })
                    
            except Exception as e:
                results.append({
                    'filename': filename,
                    'extracted_date': None,
                    'status': f'error: {str(e)}'
                })
    
    # Create DataFrame and sort by extracted date
    df = pd.DataFrame(results)
    return df

def main():
    # Specify your folder path here
    folder_path = "dataset//"
    
    # Process all PDFs in the folder
    results_df = process_pdf_folder(folder_path)
    
    # Save results to CSV
    results_df.to_excel('pdf_dates_extracted.xlsx', index=False)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total files processed: {len(results_df)}")
    print(f"Files with dates found: {len(results_df[results_df['status'] == 'success'])}")
    print(f"Files without dates: {len(results_df[results_df['status'] == 'no date found'])}")
    print(f"Files with errors: {len(results_df[results_df['status'].str.startswith('error')])}")
    
    # Display results
    print("\nResults Preview:")
    print(results_df)

if __name__ == "__main__":
    main()