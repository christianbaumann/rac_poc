import chromadb
import json
import ollama


# TODO Load collection name from file
# TODO Make usable with WebUI
# TODO Offer which collection/ book to load from
# TODO Load all collections?

# Load model name from config file
def load_model_name_from_config(config_file="model.config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config.get("model_name", "llama3:8b")  # Default model if not found in config


# Initialize ChromaDB client and retrieve the collection
def initialize_chromadb_client(path='db', collection_name="howcanitestthis"):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    return collection


# Query ChromaDB and return combined context from documents
def query_chromadb(collection, query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    print(f"Query results: {results}")

    # Combine all the document snippets into a single context
    if results['documents']:
        context = " ".join([" ".join(doc) for doc in results['documents']])
    else:
        context = "No relevant documents found."

    return context



# Construct the prompt for the LLM
def construct_ollama_prompt(context, query):
    prompt = f"Context: {context}\nAnswer the query: {query}"
    print(f"Generated prompt for Ollama: {prompt}")
    return prompt


# Main function to handle the RAG flow
def query_db_and_ollama():
    # Initialize the ChromaDB client and collection
    collection = initialize_chromadb_client()

    # Capture the user’s query
    query = input("Enter your query: ")

    # Query the collection and get the context
    context = query_chromadb(collection, query)

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
