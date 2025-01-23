from pinecone import Pinecone
import os
"""
on query, send a request to pinecone, returns the top 3 documents with a match, as well as a link to them (ID)
"""

#initialise Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index("datathon")

def lambda_handler(event, context):
    link = event['queryStringParameters']['link']
    #send query to pinecone of the link as an id, pinecone fetches the vector, gets the 3 related articles, and returns them.
    
    #fetch the vector, then fetch the 3 most similar documents to the vector. return the objects
    vector = index.fetch(link)["vectors"]
    print(vector)

    #query for the 3 most similar articles
    similar = index.query(
        vector = vector,
        top_k=3,
    )

    print(res['id'] for res in similar['matches'])
    
    return {
        "statusCode": 200,
        "body": {
            "matches": [res['id'] for res in similar['matches']]
        }
    }
