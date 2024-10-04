import chromadb
import ollama

# Load the ChromaDB and query it
def query_db_and_ollama():
    # Initialize ChromaDB client instead of Chroma
    client = chromadb.PersistentClient(path='db')  # Load from persistent storage

    # Recreate the collection
    collection = client.get_collection("default_collection")

    # Take user query
    query = input("Enter your query: ")

    # Perform a query against the indexed data
    results = collection.query(query_texts=[query], n_results=5)
    context = " ".join([result['text'] for result in results['documents']])

    # Query Ollama API with context
    prompt = f"Context: {context}\nAnswer the query: {query}"
    response = ollama.complete(prompt=prompt)

    print(f"Response from Ollama: {response}")

if __name__ == "__main__":
    query_db_and_ollama()
