import chromadb

def delete_all_collections():
    # Initialize the ChromaDB client and connect to persistent storage
    client = chromadb.PersistentClient(path='db')

    # Retrieve all collections
    collections = client.list_collections()
    if not collections:
        print("No collections found in the database.")
        return

    # Loop through and delete each collection
    for collection in collections:
        collection_name = collection.name
        print(f"Deleting collection: {collection_name}")
        client.delete_collection(collection_name)

    print("All collections have been deleted.")

if __name__ == "__main__":
    delete_all_collections()
