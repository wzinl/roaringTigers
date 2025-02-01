import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from pinecone import Pinecone
import math
import psycopg
from psycopg_pool import ConnectionPool

# AWS RDS and Pinecone Configuration
AWS_REGION = st.secrets["REGION"]   
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX = st.secrets["INDEX"]

#Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX)

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
connect_pool = ConnectionPool(conninfo=conn_str, min_size=1, max_size=10)

def get_article_entity(field, uuid):
    """Fetch documents from RDS based on a specific field."""
    with connect_pool as pool:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                if field == "gpe":
                    query = f"""SELECT g.name
                                FROM articles a
                                JOIN article_gpe ag ON a.article_id = ag.article_id
                                JOIN gpe g ON ag.gpe_id = g.gpe_id
                                WHERE a.uuid = %s;"""
                else:
                    query = f"SELECT {field} FROM ARTICLES WHERE uuid = %s;"
                cur.execute(query, (uuid,))
                return cur.fetchall()

def get_filtered_article(field, query_type):
    """Fetch documents from RDS based on a specific field."""
    with connect_pool as pool:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                if query_type == "Geo-Political Entity":
                    query = f"""SELECT *
                                FROM articles a
                                JOIN article_gpe ag ON a.article_id = ag.article_id
                                JOIN gpe g ON ag.gpe_id = g.gpe_id
                                WHERE name text = {field};"""
                elif query_type == "Tags": 
                    query = f"""SELECT * FROM ARTICLES WHERE '{field}' = ANY(zeroshot_labels);""" 
                elif query_type == "Person":
                    query = f"""SELECT * FROM ARTICLES WHERE '{field}' = ANY(persons);"""
                elif query_type == "Organisation": 
                    query = f"""SELECT * FROM ARTICLES WHERE '{field}' = ANY(orgs);"""

                cur.execute(query)
                return cur.fetchall()


def query_pinecone(doc_UUID):
    similar = index.query(
        id = doc_UUID,
        top_k = 4,
    )
    
    res_arr = [res['id'] for res in similar['matches'][1:]]
    return res_arr
        

def get_total_articles():
    """Get total number of articles in the database."""
    with connect_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM articles")
            return cur.fetchone()[0]
        
def get_paginated_articles(offset, limit):
    """Get a page of articles with basic information."""
    with connect_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uuid, title, date, summary
                FROM articles 
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            return cur.fetchall()

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

# def fetch_documents_by_field(field_type, field_value):
#     """Fetch documents from RDS based on a specific field."""
#     engine = get_rds_connection()
#     if not engine:
#         return []
    
#     query = f"""
#     SELECT * FROM articles 
#     WHERE {field_type} = '{field_value}'
#     """
#     return pd.read_sql(query, engine)

def get_paginated_articles(offset, limit):
    """Get a page of articles with basic information."""
    with connect_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uuid, title, date, summary
                FROM articles 
                ORDER BY date DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            return cur.fetchall()

def get_article_details(uuid):
    """Get detailed information about an article including entities."""
    details = {}
    
    # Get basic article information
    with connect_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, date, summary, zeroshot_labels
                FROM articles 
                WHERE uuid = %s
                """,
                (uuid,)
            )
            article = cur.fetchone()
            if article:
                title, date, summary, labels = article
                details = {
                    "title": title,
                    "date": date,
                    "summary": summary,
                    "labels": labels
                }
    
    # Get GPE entities
    details["locations"] = [row[0] for row in get_article_entity("gpe", uuid)]
    
    return details

def show_article_list():
    """Display the main article listing page."""
    st.title("Article Browser")
    
    # Pagination settings
    articles_per_page = 10
    total_articles = get_total_articles()
    total_pages = math.ceil(total_articles / articles_per_page)
    
    # Get current page from session state or sidebar
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    current_page = st.sidebar.number_input(
        "Page", 
        min_value=1, 
        max_value=total_pages, 
        value=st.session_state.current_page
    )
    st.session_state.current_page = current_page
    
    # Calculate offset for pagination
    offset = (current_page - 1) * articles_per_page
    
    # Get articles for current page
    articles = get_paginated_articles(offset, articles_per_page)
    
    # Display articles
    for uuid, title, date, summary in articles:
        with st.container():
            st.write(f"**Date: {date}**")
            if st.button(f"{title}", key=f"btn_{uuid}"):
                st.session_state.selected_article = uuid
                st.session_state.page = "article_detail"
                #st.experimental_rerun()
            #st.write(summary)
            st.divider()
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_page > 1:
            if st.button("← Previous"):
                st.session_state.current_page = current_page - 1
                #st.experimental_rerun()
    
    with col2:
        st.write(f"Page {current_page} of {total_pages}")
    
    with col3:
        if current_page < total_pages:
            if st.button("Next →"):
                st.session_state.current_page = current_page + 1
                # st.experimental_rerun()

def show_article_detail():
    """Display the article detail page."""
    details = get_article_details(st.session_state.selected_article)
    
    # Add back button
    if st.button("← Back to Articles"):
        st.session_state.page = "list"
        #st.experimental_rerun()
    
    # Display article details
    st.title(details['title'])
    st.write(f"**Date:** {details['date']}")
    
    st.header("Summary")
    st.write(details['summary'])
    
    if details['labels']:
        st.header("Labels")
        st.write(", ".join(details['labels']))
    
    if details['locations']:
        st.header("Locations Mentioned")
        st.write(", ".join(details['locations']))

def main():
    st.title("Advanced Knowledge Graph Explorer")

    # Sidebar filters
    st.sidebar.header("Filters")
        
    # Sidebar for navigation
    menu = ["Search", "Knowledge Graph"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    # Load spaCy model
    nlp = load_spacy_model()

    if choice == "Search":
        st.subheader("Document Search")
        # Search options
        search_type = st.selectbox("Search By", ["Tags", "Person", "Geo-Political Entity", "Organisation"])
        if search_type == "Tags":
            search_query = st.selectbox(f"Enter {search_type}",("legal frameworks","economic sanctions","social justice","human rights violations","global governance","governance reform","language preservation","conflict resolution","epidemic management","innovation ecosystems","global economic policy","sustainable development","institutional transparency","pandemic response","digital transformation","economic development","geopolitical tensions","cultural exchange","emerging markets","international trade","indigenous rights","regional stability","cybersecurity","international sanctions","international conflict","technological policy","healthcare access","global health policy","tech diplomacy","gender equality","green technology","poverty alleviation","climate change policy","diplomatic negotiations","humanitarian crisis","UN diplomacy","healthcare infrastructure","international security","environmental justice","vaccine distribution","military interventions","refugee policy","technology transfer","academic development","natural disaster response","conservation efforts","educational policy","environmental protection","arms control","infrastructure development","terrorism","international law","humanitarian aid","peacekeeping operations","medical research"))
        else:
            search_query = st.text_input(f"Enter {search_type}")
        
        if st.button("Search"):
            # Fetch documents
            # results = fetch_documents_by_field(search_type.lower(), search_query)
            results = get_filtered_article(search_query, search_type)

            if len(results) != 0:
                st.dataframe(results)
                
                # Related Articles from Pinecone
                related_articles = query_pinecone(search_query)
                
                st.subheader("Related Articles")
                # displaying of the 3 articles
                cols = st.columns(3)  # 3 columns for 3 articles
                for i, col in enumerate(cols):
                    with col:
                        st.subheader(get_article_entity("title", related_articles[i]))
                        date = f"📅 **Date:** {get_article_entity("date", related_articles[i])}"
                        st.write(date)
                        tags = f"🏷 **Tags:** {', '.join(tag for tag in get_article_entity("zeroshot_labels", related_articles[i]))}"
                        st.write(tags)
                        persons = f"🧑‍⚖️ **Persons:** {', '.join(entity for entity in get_article_entity("persons", related_articles[i]))}"
                        st.write(persons)
                        companies = f"🧑‍⚖️ **Organisations:** {', '.join(entity for entity in get_article_entity("orgs", related_articles[i]))}"
                        st.write(companies)

                # Knowledge Graph
                all_entities = []
                all_relations = []
                for article in related_articles:
                    entities, relations = process_text(article, nlp)
                    all_entities.append(entities)
                    all_relations.append(relations)
                G = create_graph(all_entities, all_relations)
                html_file = visualize_graph(G)
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=800)
            
                # Clean up
                os.unlink(html_file)
                
    if 'page' not in st.session_state:
        st.session_state.page = "list"
    
    # Show appropriate page based on state
    if st.session_state.page == "list":
        show_article_list()
    elif st.session_state.page == "article_detail":
        show_article_detail()
    
                
                # for article in related_articles:
                #     st.write(article['metadata']['title'])
    
    # elif choice == "Knowledge Graph":
    #     st.subheader("Knowledge Graph Visualization")
        
    #     # Input for graph generation
    #     input_text = st.text_area("Enter text to generate knowledge graph")
        
    #     if st.button("Generate Graph"):
    #         # Process text and create graph
    #         entities, relations = process_text(input_text, nlp)
    #         G = create_graph(entities, relations)
            
    #         # Visualize graph
    #         html_file = visualize_graph(G)
            
    #         with open(html_file, 'r', encoding='utf-8') as f:
    #             html_data = f.read()
    #         st.components.v1.html(html_data, height=800)
            
    #         # Clean up
    #         os.unlink(html_file)


if __name__ == "__main__":
    main()