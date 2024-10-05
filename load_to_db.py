import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import chromadb
import os
import re


# TODO Save collection name and book title to file

def generate_valid_collection_name(epub_path):
    file_name = os.path.basename(epub_path)
    base_name, _ = os.path.splitext(file_name)
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    return valid_name[:63]


# Load EPUB content and extract the title for indexing
def load_epub(file_path):
    book = epub.read_epub(file_path)
    text_content = ''
    title = book.get_metadata('DC', 'title')[0][0]  # Fetch the title from EPUB metadata

    # Concatenate the text content from document items in the EPUB
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse the content with BeautifulSoup to strip away HTML tags
            soup = BeautifulSoup(item.get_content().decode('utf-8'), 'html.parser')
            text_content += soup.get_text()  # Extract just the textual content

    # Clean up the text by removing excessive newlines and spaces
    text_content = re.sub(r'\n\s*\n+', '\n\n', text_content.strip())  # Remove excessive newlines and spaces

    # Split text into paragraphs or smaller chunks to improve indexing
    text_chunks = text_content.split("\n\n")  # Split based on double newlines (paragraphs)

    # Log the extracted and cleaned text
    print(f"Extracted text chunks (first 5 chunks): {text_chunks[:5]}")

    return text_chunks, title


# Directly interact with ChromaDB to ensure proper document indexing
def save_to_db(epub_path):
    text_chunks, title = load_epub(epub_path)

    if len(text_chunks) == 0:
        print("Error: No meaningful content to index.")
        return

    # Initialize the ChromaDB client and create a valid collection name
    vectorstore = chromadb.PersistentClient(path='db')
    collection_name = generate_valid_collection_name(epub_path)

    # Create a collection with the sanitized name in ChromaDB
    collection = vectorstore.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    # Initialize HuggingFace embeddings
    embedding_func = embedding_functions.EmbeddingFunction(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_query
    )

    # Directly add documents to the collection with embeddings
    for idx, chunk in enumerate(text_chunks):
        # Create a unique ID for each document
        doc_id = f"doc_{idx}"
        print(f"Indexing document {idx}: {chunk[:100]}")  # Print first 100 characters of each chunk
        collection.add(
            ids=[doc_id],  # Unique ID for each chunk
            documents=[chunk],  # The document content (chunk)
            embeddings=[embedding_func(chunk)]  # Embedding of the chunk
        )

    # Check how many documents were indexed after indexing
    documents_in_collection = collection.count()
    print(f"Number of documents indexed: {documents_in_collection}")

    # List all collections to verify
    collections = vectorstore.list_collections()
    print(f"All collections: {[col.name for col in collections]}")

    # Confirm success
    print(f"EPUB content titled '{title}' has been indexed with collection name: '{collection_name}'.")


if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'
    save_to_db(epub_path)
