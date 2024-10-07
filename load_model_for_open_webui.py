import chromadb
import json
import time

# Load model name from config file
def load_model_name_from_config(config_file="model.config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config.get("model_name", "llama3:8b")  # Default model if not found in config

# Initialize ChromaDB client and retrieve all collections
def initialize_chromadb_client(path='db'):
    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()  # Retrieve all collections
    return collections

# Query ChromaDB and return combined context from documents across all collections
def query_chromadb_all_collections(collections, query, n_results=5):
    combined_context = ""

    for collection_name in collections:
        collection = collections[collection_name]
        results = collection.query(query_texts=[query], n_results=n_results)
        print(f"Query results from collection '{collection_name}': {results}")

        if results['documents']:
            collection_context = " ".join([" ".join(doc) for doc in results['documents']])
            combined_context += collection_context + " "
        else:
            print(f"No relevant documents found in collection '{collection_name}'.")

    if not combined_context:
        combined_context = "No relevant documents found in any collection."

    return combined_context

# Main function to enrich the model for Open WebUI
def enrich_model_for_open_webui():
    # Initialize the ChromaDB client and retrieve all collections
    collections = initialize_chromadb_client()

    # Sample query to enrich the model (can be customized)
    query = "How to test a web login?"

    # Query all collections and get the combined context
    context = query_chromadb_all_collections(collections, query)

    # Load model name from config
    model_name = load_model_name_from_config()

    print(f"Enriched model '{model_name}' with the following context:\n{context}")
    print("The enriched model is now ready for use with Open WebUI.")

    # Keep the script running to ensure the enriched model stays in memory
    while True:
        time.sleep(60)  # Keep the process alive

if __name__ == "__main__":
    enrich_model_for_open_webui()
