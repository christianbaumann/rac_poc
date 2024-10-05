import chromadb
import json
import ollama


# TODO Load collection name from file
# TODO Make usable with WebUI

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

        # Combine all the document snippets from this collection into a single context
        if results['documents']:
            collection_context = " ".join([" ".join(doc) for doc in results['documents']])
            combined_context += collection_context + " "
        else:
            print(f"No relevant documents found in collection '{collection_name}'.")

    if not combined_context:
        combined_context = "No relevant documents found in any collection."

    return combined_context


# Construct the prompt for the LLM
def construct_ollama_prompt(context, query):
    prompt = f"Context: {context}\nAnswer the query: {query}"
    print(f"Generated prompt for Ollama: {prompt}")
    return prompt


# Main function to handle the RAG flow
def query_db_and_ollama():
    # Initialize the ChromaDB client and retrieve all collections
    collections = initialize_chromadb_client()

    # Capture the user’s query
    query = input("Enter your query: ")

    # Query all collections and get the combined context
    context = query_chromadb_all_collections(collections, query)

    # Construct the prompt for Ollama
    prompt = construct_ollama_prompt(context, query)

    # Load model name from config
    model_name = load_model_name_from_config()

    # Send the prompt to Ollama and receive the response
    response = ollama.chat(model=model_name, messages=[{"role": "system", "content": prompt}])

    # Display the content of the response
    print(f"Response from Ollama: {response['message']['content']}")


if __name__ == "__main__":
    query_db_and_ollama()
