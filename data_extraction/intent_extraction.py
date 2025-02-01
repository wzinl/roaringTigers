import spacy
import pandas as pd
from typing import Dict, Tuple
from thefuzz import fuzz
from thefuzz import process


class IntentExtractor:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Define intent keywords
        self.intent_keywords = {
    # Crime-related intents
    'REPORT_CRIME': ['arrest', 'robbery', 'fraud', 'police', 'theft', 'investigation'],
    'INVESTIGATION': ['investigate', 'probe', 'case', 'suspect', 'evidence'],
    'LEGAL_ACTION': ['charged', 'trial', 'court', 'conviction', 'lawsuit'],

    # Business-related intents
    'ANNOUNCE_BUSINESS_ACTIVITY': ['launch', 'deal', 'merger', 'acquisition', 'startup'],
    'FINANCIAL_UPDATES': ['revenue', 'profit', 'growth', 'sales', 'loss'],
    'INVESTMENT_OPPORTUNITIES': ['investment', 'venture', 'funding', 'capital', 'shares'],

    # Finance-related intents
    'ECONOMIC_ANALYSIS': ['inflation', 'forecast', 'market', 'analysis', 'economy'],
    'POLICY_CHANGES': ['interest rates', 'regulation', 'policy', 'reform', 'monetary'],
    'MARKET_UPDATES': ['stocks', 'commodities', 'oil prices', 'market trends'],

    # Politics-related intents
    'CAMPAIGN_UPDATES': ['election', 'vote', 'campaign', 'candidate', 'manifesto'],
    'POLICY_DEBATE': ['reform', 'policy', 'tax', 'debate', 'amendment'],
    'PROTEST_NEWS': ['protest', 'march', 'rally', 'demonstration', 'strike'],

    # Health-related intents
    'HEALTH_ADVISORY': ['flu', 'virus', 'outbreak', 'warning', 'vaccination'],
    'MEDICAL_RESEARCH_UPDATES': ['study', 'treatment', 'findings', 'discovery', 'trial'],
    'PANDEMIC_NEWS': ['pandemic', 'COVID-19', 'lockdown', 'cases', 'quarantine'],

    # Sports-related intents
    'EVENT_ANNOUNCEMENT': ['tournament', 'championship', 'game', 'match', 'olympics'],
    'PLAYER_UPDATES': ['transfer', 'player', 'injury', 'performance', 'goal'],
    'MATCH_RESULTS': ['win', 'defeat', 'draw', 'score', 'result'],

    # Everyday news intents
    'COMMUNITY_ANNOUNCEMENTS': ['local', 'community', 'event', 'market', 'festival'],
    'WEATHER_UPDATES': ['rain', 'forecast', 'storm', 'temperature', 'snow'],
    'HUMAN_INTEREST_STORIES': ['rescue', 'kindness', 'achievement', 'project', 'initiative'],

    # Technology-related intents
    'PRODUCT_LAUNCHES': ['product', 'AI', 'tool', 'software', 'release'],
    'RESEARCH_BREAKTHROUGHS': ['discovery', 'breakthrough', 'study', 'technology', 'innovation'],
    'CYBERSECURITY_ALERTS': ['hack', 'breach', 'data', 'security', 'attack'],

    # Entertainment-related intents
    'EVENT_ANNOUNCEMENT_ENT': ['concert', 'movie', 'show', 'release', 'event'],
    'CELEBRITY_UPDATES': ['actor', 'celebrity', 'award', 'scandal', 'rumor'],
    'REVIEWS_AND_OPINIONS': ['review', 'rating', 'opinion', 'critics', 'feedback'],

    # Environment-related intents
    'CONSERVATION_EFFORTS': ['wildlife', 'sanctuary', 'conservation', 'forest', 'species'],
    'CLIMATE_UPDATES': ['heatwave', 'global warming', 'climate', 'drought', 'flood'],
    'SUSTAINABILITY_PRACTICES': ['renewable', 'recycle', 'sustainable', 'green', 'eco-friendly'],

    # Education-related intents
    'ADMISSIONS_AND_SCHOLARSHIPS': ['scholarship', 'admission', 'opportunity', 'school', 'college'],
    'RESEARCH_FINDINGS': ['study', 'findings', 'discovery', 'research', 'academic'],
    'EXAM_UPDATES': ['exam', 'results', 'grades', 'assessment', 'tests'],

    # Conflict-related intents
    'MILITARY_UPDATES': ['troops', 'attack', 'bombing', 'war', 'missile'],
    'CEASEFIRE_AGREEMENTS': ['peace', 'agreement', 'ceasefire', 'negotiation', 'truce'],
    'PROTESTS_AND_TENSIONS': ['protest', 'strike', 'conflict', 'unrest', 'demonstration'],

    # Economy-related intents
    'ECONOMIC_REPORTS': ['GDP', 'growth', 'inflation', 'trade', 'economy'],
    'POLICY_ANNOUNCEMENTS': ['reform', 'policy', 'budget', 'tax', 'law'],
    'MARKET_TRENDS': ['stocks', 'commodities', 'prices', 'trade', 'market'],

    # Legal-related intents
    'COURT_CASES': ['trial', 'lawsuit', 'hearing', 'court', 'judgment'],
    'LAW_REFORMS': ['reform', 'bill', 'act', 'law', 'amendment'],
    'DISPUTES': ['conflict', 'dispute', 'lawsuit', 'negotiation', 'settlement']
}

        
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the text based on keywords."""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            # Score intents based on keyword matches
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        # Return the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent  # (intent, confidence)
        
        return ('UNCLASSIFIED', 0.0)
    
    def process_text(self, text: str) -> Dict:
        """Process a single text entry and extract intents."""
        doc = self.nlp(text)
        entities = {ent.label_: [] for ent in doc.ents}
        
        # Extract entities
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        return {
            'Text': text,
            'Intent': intent,
            'Confidence': confidence,
            'Entities': entities
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process a DataFrame and extract intents and entities."""
        results = []
        
        for text in df[text_column]:
            result = self.process_text(text)
            results.append(result)
        
        # Flatten entities into columns
        flat_results = []
        for result in results:
            flat_result = {
                'Text': result['Text'],
                'Intent': result['Intent'],
                'Confidence': result['Confidence'],
            }
            for entity_type, entity_list in result['Entities'].items():
                flat_result[entity_type] = ', '.join(entity_list)
            flat_results.append(flat_result)
        
        return pd.DataFrame(flat_results)

# Main execution
if __name__ == "__main__":
    input_file = r"C:\Users\easan.s.2024\Documents\news_excerpts_parsed.xlsx"  # Input file path
    output_file = 'intent_extraction_results_2.xlsx'  # Output file path
    
    # Load input data
    try:
        data = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Ensure the file is in the working directory.")
        exit()

    # Check if required column exists
    if 'Text' not in data.columns:
        print("Input file must have a 'Text' column containing text summaries.")
        exit()
    
    # Initialize the extractor
    extractor = IntentExtractor()
    
    # Process the data
    results_df = extractor.process_dataframe(data, 'Text')
    
    # Save results to Excel
    results_df.to_excel(output_file, index=False)
    print(f"Intent extraction complete. Results saved to {output_file}.")
