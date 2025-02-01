import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_date_from_text(text):
    """
    Extract date from text using various common date formats.
    """
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    return None

def extract_date_from_meta_tags(soup):
    """
    Extract date from common meta tags used by news websites.
    """
    meta_tags = [
        {'property': 'article:published_time'},
        {'property': 'article:modified_time'},
        {'name': 'date'},
        {'name': 'pubdate'},
        {'name': 'publishdate'},
        {'name': 'timestamp'},
        {'itemprop': 'datePublished'},
        {'itemprop': 'dateModified'},
        {'name': 'DC.date.issued'},
        {'name': 'article:published_time'}
    ]
    
    for tags in meta_tags:
        meta = soup.find('meta', tags)
        if meta and meta.get('content'):
            return meta['content']
    
    # Check LD+JSON
    script = soup.find('script', {'type': 'application/ld+json'})
    if script:
        try:
            import json
            data = json.loads(script.string)
            if isinstance(data, dict):
                date = data.get('datePublished') or data.get('dateModified')
                if date:
                    return date
        except:
            pass
    
    return None

def process_url(url):
    """
    Extract date from a news article URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try meta tags first
        date = extract_date_from_meta_tags(soup)
        
        # If no date in meta tags, try extracting from text
        if not date:
            article = soup.find('article') or soup.find('main') or soup.find('body')
            if article:
                article_text = article.get_text()
                date = extract_date_from_text(article_text)
        
        return {
            'url': url,
            'extracted_date': date,
            'status': 'success' if date else 'no date found',
            'type': 'web'
        }
        
    except Exception as e:
        return {
            'url': url,
            'extracted_date': None,
            'status': f'error: {str(e)}',
            'type': 'web'
        }

def process_urls_with_progress(urls):
    """
    Process URLs with a progress bar.
    """
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_url, url) for url in urls]
        for future in tqdm(futures, total=len(urls), desc="Processing URLs"):
            results.append(future.result())
    return results

def main():
    # Read URLs from Excel file
    excel_file = "path/to/your/excel_file.xlsx"
    df = pd.read_excel(excel_file)
    
    # Ensure the 'Link' column exists
    if 'Link' not in df.columns:
        raise ValueError("Excel file must contain a 'Link' column")
    
    # Get list of URLs
    urls = df['Link'].dropna().tolist()
    print(f"\nFound {len(urls)} URLs to process")
    
    # Process URLs with progress bar
    results = process_urls_with_progress(urls)
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('url_dates_extracted.csv', index=False)
    
    # Print summary
    print("\nURL Processing Summary:")
    print(f"Total URLs processed: {len(results_df)}")
    print(f"URLs with dates found: {len(results_df[results_df['status'] == 'success'])}")
    print(f"URLs without dates: {len(results_df[results_df['status'] == 'no date found'])}")
    print(f"URLs with errors: {len(results_df[results_df['status'].str.startswith('error')])}")

if __name__ == "__main__":
    main()
