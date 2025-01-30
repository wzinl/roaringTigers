import requests
import hashlib

def generate_hash_id(text):
    """Generate a unique ID using a hash function."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def req(hash):
    API_URL = ""
    params = {
        "id" : hash,
    }
    r = requests.get(API_URL, params = params)
    print(r.json())

def main():
    LINK = r"" #Enter link here
    
    hash = generate_hash_id(LINK)
    print(hash)
    req(hash)

if __name__ == "__main__":
    main()
