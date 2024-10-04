# RAG PoC

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using an EPUB ebook, LangChain, ChromaDB, and Ollama for document-based question answering.

## Setup

1. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the EPUB you want to use and place it in the `ebooks/` folder.

## Steps

### 1. Load EPUB into the Database

Run the following command to index the EPUB into ChromaDB:
```bash
python load_to_db.py
```

### 2. Query with Ollama

Use this script to interact with the database and query via Ollama:
```bash
python query_ollama.py
```

## Requirements

- Python 3.x
- See `requirements.txt` for dependencies.
