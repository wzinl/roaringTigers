import pandas as pd
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from tqdm import tqdm  # Added for progress bar

# Load the Excel file
input_file = 'dataset/news_excerpts_parsed.xlsx'
data = pd.read_excel(input_file)

# Extract relevant columns
text_column = data['Text']
link_column = data['Link']

# Initialize the tokenizer and model for multilingual-e5-large
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

def generate_embedding(text):
    """Generate embeddings using the multilingual-e5-large model."""
    input_text = f"passage: {text}"
    inputs = tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
   
    with torch.no_grad():
        outputs = model(**inputs)
   
    embeddings = outputs.last_hidden_state.masked_fill(~inputs['attention_mask'].bool().unsqueeze(-1), 0.0)
    embeddings = embeddings.sum(dim=1) / inputs['attention_mask'].sum(dim=1, keepdim=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
   
    return embeddings.squeeze().tolist()

# Initialize Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index configuration
index_name = "datathon"
dimension = 1024  # Adjust based on multilingual-e5-large model dimension

# Connect to the Pinecone index
index = pc.Index(index_name)

# Batch upload configuration
batch_size = 100
embeddings_to_upsert = []

# Use tqdm to wrap the iteration and show progress
print("Starting embedding generation and Pinecone upload...")
for text, link in tqdm(zip(text_column, link_column), total=len(text_column), desc="Generating Embeddings"):
    if pd.notna(text) and pd.notna(link):
        embedding = generate_embedding(text)
        embeddings_to_upsert.append((link, embedding))
       
        # Upsert in batches
        if len(embeddings_to_upsert) >= batch_size:
            index.upsert(embeddings_to_upsert)
            embeddings_to_upsert = []

# Upsert any remaining embeddings
if embeddings_to_upsert:
    index.upsert(embeddings_to_upsert)

print("Embedding generation and upload to Pinecone completed.")

