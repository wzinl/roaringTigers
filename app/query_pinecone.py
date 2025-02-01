from pinecone import Pinecone
import streamlit as st

#initialise Pinecone  //just add these as global variables
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
INDEX = st.secrets['INDEX']
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index(INDEX)

def query_pinecone(doc_UUID):
    similar = index.query(
        id = doc_UUID,
        top_k = 4,
    )
    
    return {
        "matches": [res['id'] for res in similar['matches'][1:]]
        #return object: [UUID1, UUID2, UUID3]
    }

