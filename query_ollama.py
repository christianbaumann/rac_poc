import chromadb
import ollama

# Load the ChromaDB and query it
def query_db_and_ollama():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path='db')  # Load from persistent storage

    # Recreate the collection
    collection = client.get_collection("howcanitestthis")

    # Take user query
    query = input("Enter your query: ")

    # Perform a query against the indexed data
    results = collection.query(query_texts=[query], n_results=5)

    # Ensure the results contain valid documents and access text
    if 'documents' in results and results['documents']:
        context = " ".join([result[0] if result else "" for result in results['documents']])
    else:
        context = "No relevant documents found."

    # Corrected the argument for Ollama API
    response = ollama.chat(model="gemma", messages=[{"role": "system", "content": prompt}])

    print(f"Response from Ollama: {response['text']}")

if __name__ == "__main__":
    query_db_and_ollama()
