import pandas as pd
import os

import uuid  # Import Python's UUID module

import psycopg

class toInsert:
    def __init__(self, input):
        
        # initialize zeroshot labels
        self.zeroshotlabels = self.extract_labels(input)

        # initialize entities in a list
        self.entities = self.get_entities(input)


    # get entities
    def get_entities(self, nerFile):
        nerdf = pd.read_excel(nerFile, sheet_name='Sheet1')
        entityList = []
        for index,row in nerdf.iterrows():
            # entities[index] = row[index]
            entities = {}
            for column, value in row.items():
                entities[column] = value
            entityList.append(entities)
        return entityList


    # Function to get the top 3 labels for each row
    def get_top_3_labels(self, row, score_columns):
        # Convert scores to numeric, coercing errors
        scores = pd.to_numeric(row[score_columns], errors="coerce")
        # Get the top 3 labels with the highest scores
        top_3_labels = scores.nlargest(3).index.tolist()
        return top_3_labels

    # Function to extract labels from the file
    def extract_labels(self, file_path):
        zeroshortdf = pd.read_excel(file_path, sheet_name='Sheet1')
        score_columns = zeroshortdf.columns[17:]  # Score columns start from index 3

        # Apply the function to each row and create a new column for the top 3 labels
        zeroshortdf['top_3_labels'] = zeroshortdf.apply(lambda row: self.get_top_3_labels(row, score_columns), axis=1)
        
        return zeroshortdf["top_3_labels"]

# Load the Excel file paths
if __name__ == "__main__":
    input_data = os.path.join("master_data", "master_sheet.xlsx")

    input_data = toInsert(input_data)

    # print(wikileaks.entities[0]['UUID'])  # Print first few rows for verification


    # Print results
    # print(newsexcerpts.zeroshotlabels)  # Print first few rows for verification
    # print(newsexcerpts.entities)  # Print first few rows for verification

    # print(wikileaks.zeroshotlabels)  # Print first few rows for verification

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        # insert password
        password="",
        host="",
        port='5432'
    ) as conn:
        with conn.cursor() as cur:
            # Execute a command: this creates a new table
            for index, labels in input_data.zeroshotlabels.items():
                label = [labels[0], labels[1], labels[2]]
                summary = input_data.entities[index]['summary']
                record_uuid = input_data.entities[index]['UUID']
                link = input_data.entities[index]['link']
                # print(labels[0])
                cur.execute("""
                    INSERT INTO public.articles(uuid, summary, zeroshot_labels, link) VALUES (%s, %s, %s, %s);
                    """, (record_uuid, summary, label, link))

