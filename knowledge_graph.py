import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Load spaCy model
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

def main():
    st.title("Knowledge Graph Builder")
    st.write("Upload an Excel file to create a knowledge graph from its text content.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        # Load spaCy model
        nlp = load_spacy_model()
        
        # Read Excel file
        try:
            df = pd.read_excel(uploaded_file, nrows = 1)
            
            # Column selector
            text_column = st.selectbox(
                "Select the column containing text data",
                df.columns
            )
            
            if st.button("Generate Knowledge Graph"):
                with st.spinner("Processing text and creating graph..."):
                    # Process all texts and collect entities and relations
                    all_entities = set()
                    all_relations = set()
                    
                    for text in df[text_column]:
                        if pd.isna(text):
                            continue
                        entities, relations = process_text(str(text), nlp)
                        all_entities.update(entities)
                        all_relations.update(relations)
                    
                    # Create and visualize graph
                    G = create_graph(all_entities, all_relations)
                    html_file = visualize_graph(G)
                    
                    # Display statistics
                    st.write(f"Found {len(all_entities)} unique entities and {len(all_relations)} relationships")
                    
                    # Display graph
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    st.components.v1.html(html_data, height=800)
                    
                    # Clean up temporary file
                    os.unlink(html_file)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()