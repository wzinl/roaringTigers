import pandas as pd
import os
import psycopg
import streamlit as st

DB_PASS = st.secrets["DB_PASS"]   



def remove_substrings(strings):
    # Sort the list by length in descending order to ensure longer strings are checked first
    strings.sort(key=len, reverse=True)
    
    result = []

    
    for string in strings:
        # Check if the current string is a substring of any string in the result list
        if not any(string in other for other in result):
            result.append(string)    
    return result

# Example usage:
if __name__ == "__main__":
    # input_file = os.path.join("master_data", "master_sheet.xlsx")
    input_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','master_data', 'master_sheet.xlsx'))

    df = pd.read_excel(input_file)
    articles = []
    for index,row in df.iterrows():
        article = {}
        article["uuid"] = row["UUID"]
        personList = row["PERSON"][1:-1].split(', ')
        for index, string in enumerate(personList):
            personList[index] = string[1:-1]
        # print(newlist)
        personList = remove_substrings(personList)
        article["updated_persons"] = personList

        orgList = row["ORG"][1:-1].split(', ')
        for index, string in enumerate(orgList):
            orgList[index] = string[1:-1]
        article["updated_orgs"] = orgList
        articles.append(article)

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        # insert password
        password=DB_PASS,
        host="datathondb.cpeue8qq0l9u.ap-southeast-1.rds.amazonaws.com",
        port='5432'
    ) as conn:
        with conn.cursor() as cur:
            for article in articles:
                if article["updated_persons"] != ['']:
                    cur.execute("""
                    UPDATE articles SET persons = %s WHERE uuid = %s;
                    """, (article["updated_persons"], article["uuid"]))
                if article["updated_orgs"] != ['']:
                    cur.execute("""
                    UPDATE articles SET orgs = %s WHERE uuid = %s;
                    """, (article["updated_orgs"], article["uuid"]))
                