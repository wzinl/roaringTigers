import streamlit as st
import streamlit.components.v1 as components
import spacy
from spacy.tokens import Token, Span
import networkx as nx
from pyvis.network import Network
import tempfile
from pinecone import Pinecone
import math
from psycopg_pool import ConnectionPool
import uuid
import os
import Levenshtein

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
    connect_pool = ConnectionPool(conninfo=conn_str, min_size=1, max_size=10)
    # Fetch documents from RDS based on a specific field.
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

def fetch_article(field ,value):
    # Fetch documents from RDS based on a specific field.
    connect_pool = ConnectionPool(conninfo=conn_str, min_size=1, max_size=10)
    with connect_pool.connection() as conn:
        with conn.cursor() as cur:
            if field == "Person":
                field = "persons"
                value = value[0].capitalize() + value[1:]
            if field == "Geo-Political Entity":
                # field = "gpe"
                field = "name"
            if field == "Organisation":
                field = "orgs"
                value = value[0].capitalize() + value[1:]
            if field == "Tags":
                field = "zeroshot_labels"
            if field == "zeroshot_labels" or field == "orgs" or field== "persons":
                query = f"""
                SELECT 
                ARRAY_AGG(g.name) AS gpe_names,
                a.title, a.orgs, a.zeroshot_labels, a.persons, a.link, a.date, a.summary
                FROM ARTICLES a
                JOIN article_gpe ag ON a.article_id = ag.article_id
                JOIN gpe g ON ag.gpe_id = g.gpe_id
                WHERE %s = ANY(a.{field})
                GROUP BY 
                    a.title, 
                    a.orgs, 
                    a.zeroshot_labels, 
                    a.persons, 
                    a.link, 
                    a.date, 
                    a.summary;
                """
                cur.execute(query, (value,))
            else:
                if field == "uuid":
                    fieldStr = f"a.{field}"
                else:    
                    fieldStr = f"g.{field}"

                query = f"""
                SELECT 
                ARRAY_AGG(g.name) AS gpe_names,
                a.title, a.orgs, a.zeroshot_labels, a.persons, a.link, a.date, a.summary
                FROM ARTICLES a
                JOIN article_gpe ag ON a.article_id = ag.article_id
                JOIN gpe g ON ag.gpe_id = g.gpe_id
                WHERE %s = {fieldStr}
                GROUP BY 
                    a.title, 
                    a.orgs, 
                    a.zeroshot_labels, 
                    a.persons, 
                    a.link, 
                    a.date, 
                    a.summary;
                """
                cur.execute(query, (value, ))
            output = []
            results = cur.fetchall()
            for result in results:
                output_dict = {"Title": result[1],"Geo-Political Entities": result[0], "Orgs": result[2], "Tags": result[3], "Persons": result[4],
                        "Link": result[5], "Date": result[6], "Summary": result[7]}
                output.append(output_dict)
            return output

def get_filtered_article(field, query_type):
    # Fetch documents from RDS based on a specific field.
    connect_pool = ConnectionPool(conninfo=conn_str, min_size=1, max_size=10)
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

def clean_entities(entities, db_ents, method = 'levenshtein'):
    matches = {}
    
    for extracted in db_ents:
        best_match = None
        
        for reference in entities:
            # Levenshtein calculates character-level edit distance
            if method == 'levenshtein':
                # Normalize by dividing by longer string length
                score = 1 - (Levenshtein.distance(extracted, reference) / 
                             max(len(extracted), len(reference)))
        
    if score > 0.7:
        matches[extracted] = {
            'match': best_match
        }
    return matches

def process_text(text: str, nlp, db_ents):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    doc = nlp(text)
    entities = []
    relations = []
    
    # Extract named entities with special handling for organizations and people
    seen_entities = set()
    for ent in doc.ents:
        # Clean entity text
        clean_text = ent.text.strip('" ')
        if clean_text and clean_text not in seen_entities:
            entities.append((clean_text, ent.label_))
            seen_entities.add(clean_text)

    def find_quoted_text(sent: Span):
        """Find quoted text and its speaker in a sentence."""
        quotes = []
        quote_start = None
        quote_text = ""
        
        for i, token in enumerate(sent):
            if token.text in ['"', '"', '"']:
                if quote_start is None:
                    quote_start = i
                else:
                    quote_text = sent[quote_start+1:i].text
                    # Find the speaker (usually before the quote)
                    speaker = None
                    for ent in sent.ents:
                        if ent.end < quote_start:
                            speaker = ent
                    if speaker:
                        quotes.append((speaker, quote_text))
                    quote_start = None
        return quotes

    def find_reporting_verbs(token: Token) -> bool:
        """Check if token is a reporting verb commonly used in news."""
        reporting_verbs = {
            'say', 'tell', 'report', 'announce', 'state', 'mention',
            'inform', 'indicate', 'reveal', 'confirm', 'add', 'note'
        }
        return token.lemma_ in reporting_verbs

    # Process each sentence
    for sent in doc.sents:
        sent_ents = list(sent.ents)
        
        if len(sent_ents) < 2:
            continue
        
        # Find quotes and their speakers
        quotes = find_quoted_text(sent)
        for speaker, quote in quotes:
            relations.append((speaker.text, "said", quote))

        
        # Process entity pairs
        for i, ent1 in enumerate(sent_ents):
            for ent2 in sent_ents[i+1:]:
                
                # Skip if entities are too far apart
                TOKEN_DISTANCE_THRESHOLD = 15  # Increased for news articles
                if abs(ent1.root.i - ent2.root.i) > TOKEN_DISTANCE_THRESHOLD:
                    continue

                relation = None
                
                # Check for reporting relationships
                for token in sent:
                    if find_reporting_verbs(token):
                        if ent1.root.head == token or any(t.head == token for t in ent1.root.children):
                            relation = token.lemma_
                            relations.append((ent1.text, relation, ent2.text))
                
                # Check for ownership/affiliation patterns
                if not relation:
                    for token in sent:
                        if token.dep_ == "prep" and token.text == "of":
                            if (ent1.end <= token.i <= ent2.start or 
                                ent2.end <= token.i <= ent1.start):
                                relations.append((ent2.text, "part_of", ent1.text))

                # Add general relationship if none found
                if not relation:
                    # Look for verb connections
                    for token in sent:
                        if (token.pos_ == "VERB" and 
                            token.i > ent1.end and token.i < ent2.start):
                            relation = token.lemma_
                            relations.append((ent1.text, relation, ent2.text))
                            break
    return entities, relations

def create_graph(entities, relations):
    # Initialize network with physics enabled
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    g = nx.Graph()
    
    # Disable force_atlas_2based as it can sometimes cause edge visibility issues
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)
    net.show_buttons(filter_=['physics'])
    
    # Entity color mapping
    entity_colors = {
        'PERSON': '#FF6B6B',
        'ORG': '#4ECDC4',
        'DATE': '#45B7D1',
        'GPE': '#96CEB4',
        'NORP': '#FFEEAD',
        'MONEY': '#D4A373',
        'PERCENT': '#A2D2FF',
        'DEFAULT': '#CBD5E1'
    }

    # for e in entities:
    #     g.add_node(e[0])
    # for r in relations:
    #     g.add_edge(r[0],r[1],label = r[2], title = r[2])
    
    # print()
    # print(g.edges(data = True))
    
    # # Add nodes directly to pyvis Network instead of using NetworkX
    added_nodes = set()
    
    # Add nodes
    for item in entities:
        entity = item[0]
        entity_type = item[1]
        node_id = str(entity).strip()
        if node_id and node_id not in added_nodes:
            color = entity_colors.get(str(entity_type), entity_colors['DEFAULT'])
            net.add_node(
                node_id,
                label=node_id,
                title=f"Type: {entity_type}",
                color=color,
                size=25,
                font={'size': 12, 'face': 'Arial'},
                shape='dot',
                borderWidth=2,
                borderWidthSelected=4
            )
            added_nodes.add(node_id)
    
    # Add edges directly
    for source, relation, target in relations:
        source_id = str(source).strip()
        target_id = str(target).strip()
        relation = str(relation).strip()
        
        # Add missing nodes if they don't exist
        for node_id in [source_id, target_id]:
            if node_id and node_id not in added_nodes:
                net.add_node(
                    node_id,
                    label=node_id,
                    title="Type: Unknown",
                    color=entity_colors['DEFAULT'],
                    size=25,
                    font={'size': 12, 'face': 'Arial'},
                    shape='dot',
                    borderWidth=2,
                    borderWidthSelected=4
                )
                added_nodes.add(node_id)
        
        # Add edge with custom styling
        if source_id and target_id:
            net.add_edge(
                source_id,
                target_id,
                title=relation,
                label=relation,
                font={'size': 10, 'face': 'Arial'},
                width=2,
                arrows='to',
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )
    return net

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
            if st.button(f"{title}", key=f"btn_{uuid}"):
                st.session_state.selected_article = uuid
                st.session_state.page = "article_detail"
            st.divider()
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_page > 1:
            if st.button("← Previous"):
                st.session_state.current_page = current_page - 1
    
    with col2:
        st.write(f"Page {current_page} of {total_pages}")
    
    with col3:
        if current_page < total_pages:
            if st.button("Next →"):
                st.session_state.current_page = current_page + 1

def show_article_detail(nlp):
    """Display the article detail page."""
    details = get_article_details(st.session_state.selected_article)
    related_articles = query_pinecone(str(st.session_state.selected_article))
    all_ents = []
    all_relations = []
    all_summaries = ""
    all_summaries += "Summary: " + str(get_article_entity("summary", st.session_state.selected_article))
    for article in related_articles:
        summary = get_article_entity("summary", uuid.UUID(article))
        all_summaries += "Summary: " + str(summary)
    
    curr_entities = fetch_article('uuid',st.session_state.selected_article)
    ents, relations = process_text(all_summaries, nlp, curr_entities)

    #call graph function
    g = create_graph(ents, relations)
    #graph_html = visualize_graph(g)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        path = tmp_file.name
    
    # Save and read the network
    g.save_graph(path)
    with open(path, 'r', encoding='utf-8') as file:
        html_string = file.read()

    # Delete temporary file
    os.unlink(path)
    
    # Display in Streamlit
    components.html(html_string, height=750, width=900, scrolling=True)


    st.subheader("Related Articles")
    #displaying of the 3 articles
    cols = st.columns(3)  # 3 columns for 3 articles
    
    for i, col in enumerate(cols):
        with col:
            result = fetch_article("uuid", related_articles[i])
            if len(result) >= 1:
                result = result[0]
                st.subheader(result["Title"])
                date = f"📅 **Date:** {result["Date"]}"
                st.write(date)
                tags = f"🏷 **Tags:** {result["Tags"]}"
                st.write(tags)
                persons = f"🧑‍⚖️ **Persons:** {result["Persons"]}"
                st.write(persons)
                companies = f"🧑‍⚖️ **Organisations:** {result["Orgs"]}"
                st.write(companies)

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
    menu = ["Search", "Knowledge Graph", "Timeline"]
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
            results = fetch_article(search_type, search_query)
            if len(results) != 0:
                st.dataframe(results)

               
    if 'page' not in st.session_state:
        st.session_state.page = "list"
    
    # Show appropriate page based on state
    if st.session_state.page == "list":
        show_article_list()
    elif st.session_state.page == "article_detail":
        show_article_detail(nlp)
    
if __name__ == "__main__":
    main()