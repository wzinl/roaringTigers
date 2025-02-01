import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import boto3
import requests
import hashlib
from pinecone import Pinecone

import psycopg
from psycopg_pool import ConnectionPool

# AWS RDS and Pinecone Configuration
AWS_REGION = st.secrets["REGION"]   
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX = st.secrets["INDEX"]



def execute_query(query, params=None):
    # Acquire a connection from the pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM your_table", params)
            if cur.description:  # If the query returns data (e.g., SELECT)
                return cur.fetchall()

# Database Connections
    # """Establish connection to AWS RDS."""
# Database configuration
# Create the connection string
conn_str = (
    f"dbname={st.secrets["DBNAME"]} "
    f"user={st.secrets["USER"]} "
    f"password={st.secrets["PASSWORD"]} "
    f"host={st.secrets["HOST"]} "
    f"port={st.secrets["PORT"]}"
)

# Initialize the connection pool
pool = ConnectionPool(conninfo=conn_str, min_size=1, max_size=10)


def get_pinecone_results(query):
    """Fetch related articles from Pinecone."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX)
    
    # Perform similarity search
    results = index.query(query, top_k=4, include_metadata=True)
    return results['matches']

# Cached spaCy model loading
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

def process_text(text, nlp):
    """Extract entities and their relationships from text."""
    doc = nlp(text)
    entities = []
    relations = []
    
    # Extract named entities
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
        
    # Create relationships between entities that appear in the same sentence
    for sent in doc.sents:
        sent_ents = [ent for ent in sent.ents]
        for i, ent1 in enumerate(sent_ents):
            for ent2 in sent_ents[i+1:]:
                relations.append((ent1.text, ent2.text))
    
    return entities, relations

def create_graph(entities, relations):
    """Create a NetworkX graph from entities and relations."""
    G = nx.Graph()
    
    # Add nodes with entity types as attributes
    for entity, entity_type in entities:
        G.add_node(entity, title=f"{entity} ({entity_type})")
    
    # Add edges
    G.add_edges_from(relations)
    
    return G

def visualize_graph(G):
    """Convert NetworkX graph to Pyvis network for visualization."""
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Copy nodes and edges from NetworkX graph to Pyvis network
    for node, node_attrs in G.nodes(data=True):
        net.add_node(node, title=node_attrs.get('title', node))
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name

def fetch_documents_by_field(field_type, field_value):
    """Fetch documents from RDS based on a specific field."""
    engine = get_rds_connection()
    if not engine:
        return []
    
    query = f"""
    SELECT * FROM articles 
    WHERE {field_type} = '{field_value}'
    """
    return pd.read_sql(query, engine)

def main():
    st.title("Advanced Knowledge Graph Explorer")
    
    # Sidebar for navigation
    menu = ["Document Upload", "Search", "Knowledge Graph"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    if choice == "Document Upload":
        st.subheader("Upload Excel Document")
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file, nrows=3)
            text_column = st.selectbox("Select text column", df.columns)
            
            if st.button("Process Document"):
                # Process and store document logic here
                pass
    
    elif choice == "Search":
        st.subheader("Document Search")
        
        # Search options
        search_type = st.selectbox("Search By", ["Tags", "Entities", "Datetime"])
        search_query = st.text_input(f"Enter {search_type}")
        
        if st.button("Search"):
            # Fetch documents
            results = fetch_documents_by_field(search_type.lower(), search_query)
            
            if not results.empty:
                st.dataframe(results)
                
                # Related Articles from Pinecone
                related_articles = get_pinecone_results(search_query)
                st.subheader("Related Articles")
                for article in related_articles:
                    st.write(article['metadata']['title'])
    
    elif choice == "Knowledge Graph":
        st.subheader("Knowledge Graph Visualization")
        
        # Input for graph generation
        input_text = st.text_area("Enter text to generate knowledge graph")
        
        if st.button("Generate Graph"):
            # Process text and create graph
            entities, relations = process_text(input_text, nlp)
            G = create_graph(entities, relations)
            
            # Visualize graph
            html_file = visualize_graph(G)
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=800)
            
            # Clean up
            os.unlink(html_file)

if __name__ == "__main__":
    main()