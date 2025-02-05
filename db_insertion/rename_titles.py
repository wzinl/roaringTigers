import pandas as pd
import re
import psycopg
import streamlit as st

DB_PASS = st.secrets["DB_PASS"]   
# Example usage:
if __name__ == "__main__":

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        # insert password
        password=DB_PASS,
        host="datathondb.cpeue8qq0l9u.ap-southeast-1.rds.amazonaws.com",
        port='5432'
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                title, uuid
                FROM ARTICLES 
            """ )
            output = []
            results = cur.fetchall()
            pattern = r"^https:\/\/www\.channelnewsasia\.com\/.*"
            for result in results:
                title = result[0]
                uuid = result[1]
                # print(title)
                if re.match(pattern, title):
                    title = title[title.rfind('/') + 1:title.rfind('-')].replace("-", " ").capitalize()
                    print(title)
                    cur.execute("""
                    UPDATE articles SET title = %s WHERE uuid = %s;
                    """, (title, uuid))
            # print(results)
                