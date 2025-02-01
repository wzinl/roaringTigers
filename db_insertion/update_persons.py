import pandas as pd
import os
import psycopg


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
        newlist = row["PERSON"][1:-1].split(', ')
        for index, string in enumerate(newlist):
            newlist[index] = string[1:-1]
        # print(newlist)
        newlist = remove_substrings(newlist)
        article["updated_persons"] = newlist
        articles.append(article)

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        # insert password
        password="",
        host="datathondb.cpeue8qq0l9u.ap-southeast-1.rds.amazonaws.com",
        port='5432'
    ) as conn:
        with conn.cursor() as cur:
            for article in articles:
                if article["updated_persons"] != ['']:
                    cur.execute("""
                    UPDATE articles SET link = %s WHERE uuid = %s;
                    """, (article["updated_persons"], article["uuid"]))