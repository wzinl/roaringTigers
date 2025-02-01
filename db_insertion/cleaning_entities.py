import pandas as pd
import os

import uuid  # Import Python's UUID module

import psycopg

class toInsert:
    def __init__(self, extracted):
        

        # initialize entities in a list
        self.entities = self.get_entities(extracted)


    # get entities
    def get_entities(self, extracted):
        nerdf = pd.read_excel(extracted, sheet_name='Sheet1')
        entityList = []
        for index,row in nerdf.iterrows():
            # entities[index] = row[index]
            entities = {}
            for column, value in row.items():
                entities[column] = value
            entityList.append(entities)
        return entityList
    


# check if one entity is an abbreviation
def is_abbreviation(str1: str, str2: str) -> bool:
    def get_abbreviation(full_str: str) -> str:
        """
        Generate an abbreviation by taking the first letter of each word.
        """
        words = full_str.split()
        return ''.join(word[0] for word in words if word).lower()

    # Convert both strings to lowercase for case-insensitive comparison
    str1_lower = str1.lower()
    str2_lower = str2.lower()

    # Check if str1 is an abbreviation of str2
    if str1_lower == get_abbreviation(str2):
        return True
    
    # Check if str2 is an abbreviation of str1

    return False

def check_indiv_entity(indiv_entity, entity_dict):
    indiv_entity_lower = indiv_entity.lower()
    already_exists = False
    updated_entity = indiv_entity_lower
    if indiv_entity_lower[0:3] == 'the':
        indiv_entity_lower = indiv_entity_lower[4:]
    if indiv_entity_lower[-1:] in ['\'', '’']:
        indiv_entity_lower = indiv_entity_lower[:-1]
    if indiv_entity_lower[-2:] in ['\'s', '’s'] :
        indiv_entity_lower = indiv_entity_lower[:-2]

    for key, value_list in entity_dict.items():
        if indiv_entity_lower == key.lower():
            updated_entity = key
            already_exists = True
            break

        if indiv_entity in value_list:
            updated_entity = key
            already_exists = True
            break

        if indiv_entity_lower in value_list:
            updated_entity = key
            if indiv_entity not in entity_dict[key]:
                entity_dict[key].append(indiv_entity)
            break
        if is_abbreviation(indiv_entity_lower, key.lower()):
            updated_entity = key
            if indiv_entity not in entity_dict[key]:
                entity_dict[key].append(indiv_entity)
            break
    if not already_exists:
        entity_dict[indiv_entity_lower.capitalize()] = []
    return updated_entity, entity_dict




def standardize_entities(extracted, entity):
    entity_dict = {"Singapore" : ["SG"],
                   "United States" : ["U.S.", "us", "united states of america", "USA"],
                   "New Zealand" : ["NZL"],
                   "United Kingdom" : ["UK"]
                   }
    output = []
    for i, article in enumerate(extracted):
        enity_list = article[entity][1:-1].split(', ')
        new_list = []
        for j, indiv_entity in enumerate(enity_list):
            indiv_entity =  enity_list[j] = indiv_entity[1:-1]
            updated_entity, entity_dict = check_indiv_entity(indiv_entity, entity_dict)
            if updated_entity != '' and updated_entity not in new_list:
                if updated_entity[:4] == 'the ':
                    updated_entity = updated_entity[4:]
                new_list.append(updated_entity)
        extracted[i]["GPE"] = str(new_list)
        output.append({"uuid": extracted[i]["UUID"], "GPE": new_list, "summary" : extracted[i]["summary"]})

    return output
            
    
    
# Load the Excel file paths
if __name__ == "__main__":
    input_data = os.path.join("master_data", "master_sheet.xlsx")
    extracted = toInsert(input_data).entities
    standardized = standardize_entities(extracted, 'GPE')

    with psycopg.connect(
        dbname="postgres",
        user="postgres",
        # insert password
        password="",
        host="",
        port='5432'
    ) as conn:
        with conn.cursor() as cur:
            for article in standardized:
                for gpe in article['GPE']:
                    cur.execute("""
                        insert into  article_gpe ( article_id, gpe_id )
                        values ( (select article_id from articles where uuid = %s),
                        (select gpe_id from gpe where name = %s) );
                        """, (article["uuid"], gpe))

