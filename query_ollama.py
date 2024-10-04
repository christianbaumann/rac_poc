import chromadb
import ollama

# Load the ChromaDB and query it
def query_db_and_ollama():
    # Load ChromaDB
    vectorstore = chromadb.Chroma(persist_directory='db')  # Load from persistent storage
    index = vectorstore.create_index()  # Recreate index from stored data

    # Take user query
    query = input("Enter your query: ")

    # Perform a query against the indexed data (you can customize the logic here)
    docs = index.similarity_search(query)
    context = " ".join([doc.page_content for doc in docs])

    # Query Ollama API with context
    prompt = f"Context: {context}\nAnswer the query: {query}"
    response = ollama.complete(prompt=prompt)

    print(f"Response from Ollama: {response}")

if __name__ == "__main__":
    query_db_and_ollama()
