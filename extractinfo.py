import spacy
import pandas as pd
from typing import Dict, Tuple
import pycountry
from thefuzz import fuzz
from thefuzz import process


class EntityExtractor:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize country name mappings
        self.country_mappings = self._initialize_country_mappings()
        
        self.labelList = list()
        print(self.nlp.get_pipe("ner").labels)
        for label in self.nlp.get_pipe("ner").labels:
            if(label in ["DATE", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]):
                self.labelList.append(label)
            
        # print(labelList)
        # Initialize event keywords for classification
        self.event_keywords = {
            'CRIME': ['arrest', 'theft', 'fraud', 'criminal', 'police', 'investigation'],
            'FINANCIAL': ['bankruptcy', 'merger', 'acquisition', 'stock', 'investment'],
            'POLITICAL': ['election', 'vote', 'campaign', 'protest', 'parliament'],
            'CONFLICT': ['war', 'attack', 'military', 'fighting', 'troops'],
            'DISASTER': ['earthquake', 'flood', 'hurricane', 'disaster', 'accident'],
            'SPORTS': ['tournament', 'championship', 'match', 'game', 'olympics'],
            'BUSINESS': ['launch', 'partnership', 'contract', 'deal', 'startup']
        }

        
    def _initialize_country_mappings(self) -> Dict[str, str]:
        """Initialize standardized country name mappings using pycountry"""
        mappings = {}
        
        for country in pycountry.countries:
            official_name = country.name
            mappings[official_name.lower()] = official_name
            if hasattr(country, 'common_name'):
                mappings[country.common_name.lower()] = official_name
            if hasattr(country, 'official_name'):
                mappings[country.official_name.lower()] = official_name
                
        custom_mappings = {
            'usa': 'United States',
            'us': 'United States',              
            'america': 'United States',
            'uk': 'United Kingdom',
            'britain': 'United Kingdom',
            'korea north': 'North Korea',
            'korea south': 'South Korea',
            'dpr korea': 'North Korea',
            'republic of korea': 'South Korea'
        }
        mappings.update(custom_mappings)
        
        return mappings

    def standardize_name(self, name: str) -> str:
        """Standardize person names to 'FirstName LastName' format"""
        name = ' '.join(name.split()).title()
        if ',' in name:
            last_name, first_name = name.split(',', 1)
            name = f"{first_name.strip()} {last_name.strip()}"
        
        titles = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sir', 'Dame', 'Madam', 'Doctor', 'Professor']
        for title in titles:
            if name.startswith(f"{title} "):
                name = name[len(title)+1:]
        
        return name.strip()

    def standardize_country(self, country: str) -> str:
        """Standardize country names using fuzzy matching"""
        if not country:
            return None
            
        country_lower = country.lower()
        
        if country_lower in self.country_mappings:
            return self.country_mappings[country_lower]
        
        best_match = process.extractOne(
            country_lower,
            self.country_mappings.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80
        )
        
        if best_match:
            return self.country_mappings[best_match[0]]
        
        return country

    def classify_event(self, text: str) -> Tuple[str, float]:
        """Classify the type of event and return confidence score"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.event_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score / len(keywords)
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category
        
        return ('UNCLASSIFIED', 0.0)

    def process_article(self, summary: str) -> Dict:
        """Process a single article summary and extract structured information"""
        doc = self.nlp(summary)
        
        # Extract and standardize names
        names = set()
        countries = set()

        entDict = {'summary': summary}
        for label in self.labelList:
            entDict[label] = set()
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                standardized_name = self.standardize_name(ent.text)
                entDict[ent.label_].add(standardized_name)
            elif ent.label_ in self.labelList:
                entDict[ent.label_].add(ent.text)
        for label in entDict:
            entDict[label] = sorted(entDict[label])
            
        entDict['summary'] =  summary
        return entDict
        # for ent in doc.ents:


        #     if ent.label_ in ['GPE', 'LOC']:
        #         standardized_country = self.standardize_country(ent.text)
        #         if standardized_country:
        #             countries.add(standardized_country)
        # # 
        # Extract and standardize countries

        
        # # Deduplicate and sort
        # names = sorted(names)  # Sorted list of unique namesx     
        # countries = sorted(countries)  # Sorted list of unique countries
        
        # # Classify event
        # event_type, confidence = self.classify_event(summary)
        
        # return {
        #     'names': names,
        #     'countries': countries,
        #     'event_type': event_type,
        #     'event_confidence': confidence,
        #     'summary': summary
        # }

    def process_dataframe(self, df: pd.DataFrame, summary_col: str) -> pd.DataFrame:
        """Process entire dataframe of articles"""
        results = []
        for summary in df[summary_col]:
            results.append(self.process_article(summary))
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    input_file = r'dataset//news_excerpts_parsed.xlsx'
    output_file = r'dataset//extracted_info.xlsx'
    
    # Load input data
    data = pd.read_excel(input_file)
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Process data
    results = extractor.process_dataframe(data, 'Text')

    #results is a dataframe. extract all lines of data
    
    # Save output data
    results.to_excel(output_file, index=False)
    print(f"Extraction complete. Results saved to {output_file}.")
