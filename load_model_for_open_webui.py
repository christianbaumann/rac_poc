import chromadb
import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

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

    for collection in collections:  # Iterate over collection objects directly
        results = collection.query(query_texts=[query], n_results=n_results)
        print(f"Query results from collection '{collection.name}': {results}")

        # Combine all the document snippets from this collection into a single context
        if results['documents']:
            collection_context = " ".join([" ".join(doc) for doc in results['documents']])
            combined_context += collection_context + " "
        else:
            print(f"No relevant documents found in collection '{collection.name}'.")

    if not combined_context:
        combined_context = "No relevant documents found in any collection."

    return combined_context

# API to handle queries from Open WebUI
@app.route("/query", methods=["POST"])
def handle_query():
    # Get the query from the WebUI request
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Initialize the ChromaDB client and retrieve all collections
    collections = initialize_chromadb_client()

    # Query all collections and get the combined context
    context = query_chromadb_all_collections(collections, query)

    # Load model name from config
    model_name = load_model_name_from_config()

    # Enrich the model for Open WebUI
    print(f"Enriched model '{model_name}' with the following context:\n{context}")

    # Return the response to WebUI
    return jsonify({
        "model_name": model_name,
        "context": context
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Runs the Flask server to listen for queries from WebUI
