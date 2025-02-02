# NewsNetwork

NewsNetwork aims to provide a comprehensive, visual link between entities extracted from a chosen PDF. By leveraging entity extraction, semantic search, and entity linking, it generates a detailed knowledge graph that connects the given article to the three most semantically similar articles. Users can filter articles based on event tags, entities, and date of publication.

## Installation

1. Move the `secrets.toml` file provided in Google Sheets to the `.streamlit` folder.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. From the `roaringTigers` (main) directory, run the following command:

   ```bash
   streamlit run app.py
   ```

## Features

- **Entity Extraction:** Identifies key entities from the uploaded PDF.
- **Semantic Search:** Finds and links the three most semantically similar articles.
- **Knowledge Graph Visualization:** Displays relationships between entities and articles.
- **Filtering Options:** Allows users to filter articles by:
  - Event tags
  - Entities
  - Date of publication
