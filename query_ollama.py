import chromadb
import ollama

# TODO Load model name from config
# TODO Load collection name from file
# TODO Offer which collection/ book to load from

# Load ChromaDB and query it with a user's input
def query_db_and_ollama():
    # Initialize the ChromaDB client and connect to persistent storage
    client = chromadb.PersistentClient(path='db')

    # Retrieve the collection that stores indexed documents
    collection = client.get_collection("howcanitestthis")

    # Capture the user’s query
    query = input("Enter your query: ")

    # Query the collection for relevant documents
    results = collection.query(query_texts=[query], n_results=5)
    # Add logging for query results
    print(f"Query results: {results}")

    # Extract document content or provide a fallback if no results are found
    context = " ".join([result[0] if result else "" for result in results['documents']]) if results[
        'documents'] else "No relevant documents found."

    # Construct the prompt for the Ollama model using the context from ChromaDB
    prompt = f"Context: {context}\nAnswer the query: {query}"

    print(f"Generated prompt for Ollama: {prompt}")

    # Send the prompt to Ollama and receive the response
    response = ollama.chat(model="llama3:8b", messages=[{"role": "system", "content": prompt}])

    # Display the content of the response
    print(f"Response from Ollama: {response['message']['content']}")


if __name__ == "__main__":
    query_db_and_ollama()
