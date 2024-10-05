# RAG PoC (Retrieval-Augmented Generation Proof of Concept)

This project demonstrates a Retrieval-Augmented Generation (RAG) approach by integrating ChromaDB with an LLM (Ollama) to answer user queries based on relevant document contexts stored in a local database.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- ChromaDB
- Ollama API

Install the required dependencies via pip:

```bash
pip install chromadb ollama
```

## Configuration for Model Selection

The application now supports dynamic model selection through a configuration file. Instead of hardcoding the model name, you can specify the model to be used by the LLM in a `model.config.json` file.

### How to Use

1. **Create a `model.config.json` File**  
   This file should be placed in the root directory of the project. It should look like this:

   ```json
   {
     "model_name": "llama3:8b"
   }
   ```

   You can replace `"llama3:8b"` with the name of the model you wish to use.

2. **Running the Script**  
   When you run the script, it will automatically load the model name from the configuration file and use it to query the LLM.

   ```bash
   python query_ollama.py
   ```

   You’ll be prompted to enter your query, and the script will process it using the model specified in `model.config.json`.

## Features

- **Dynamic Model Loading**: The model used for the LLM can be configured in a JSON file (`model.config.json`), allowing you to switch between different models without changing the code.
- **Context Retrieval from ChromaDB**: The system queries ChromaDB for document snippets that are relevant to the user’s query and uses them to build a context for the LLM.
- **LLM Interaction**: The system constructs a prompt from the retrieved context and sends it to the LLM for generating answers.

## TODO List

This project is still under development, and here are the features to be implemented:

- Load collection names from a file.
- Make the system usable with a WebUI for easier interaction.
- Offer the ability to select which collection or book to load from.
- Load all collections and aggregate context from them.

## Code Overview

Here’s a brief explanation of the key components of the code:

### `load_model_name_from_config`

This function loads the model name from a JSON configuration file (`model.config.json`). If the file doesn’t exist or the model name isn’t provided, it falls back to the default model (`llama3:8b`).

```python
def load_model_name_from_config(config_file="model.config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config.get("model_name", "llama3:8b")  # Default model if not found in config
```

### `initialize_chromadb_client`

This function initializes the ChromaDB client and retrieves a collection from the local database:

```python
def initialize_chromadb_client(path='db', collection_name="howcanitestthis"):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    return collection
```

### `query_chromadb`

This function queries a specific collection in ChromaDB and returns the combined context from the relevant documents:

```python
def query_chromadb(collection, query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['documents']:
        context = " ".join([" ".join(doc) for doc in results['documents']])
    else:
        context = "No relevant documents found."
    return context
```

### `construct_ollama_prompt`

This function constructs a prompt for the LLM using the retrieved context and user query:

```python
def construct_ollama_prompt(context, query):
    prompt = f"Context: {context}\nAnswer the query: {query}"
    return prompt
```

### `query_db_and_ollama`

This is the main function that ties everything together. It retrieves document context from ChromaDB and sends it to the LLM for generating a response:

```python
def query_db_and_ollama():
    collection = initialize_chromadb_client()
    query = input("Enter your query: ")
    context = query_chromadb(collection, query)
    prompt = construct_ollama_prompt(context, query)
    model_name = load_model_name_from_config()
    response = ollama.chat(model=model_name, messages=[{"role": "system", "content": prompt}])
    print(f"Response from Ollama: {response['message']['content']}")
```
