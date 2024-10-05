import os
import re
import warnings
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Constants for configuration
DB_PATH = 'db'
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPUB_FILE_PATH = 'ebooks/howcanitestthis.epub'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_valid_collection_name(epub_path):
    file_name = os.path.basename(epub_path)
    base_name, _ = os.path.splitext(file_name)
    return re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)[:63]

def extract_text_from_epub(file_path):
    """Extracts raw text content and title from the EPUB file."""
    book = epub.read_epub(file_path)
    text_content = ''
    title = book.get_metadata('DC', 'title')[0][0]

    # Extract text content from EPUB
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content().decode('utf-8'), 'html.parser')
            text_content += soup.get_text()

    return text_content, title

def clean_and_split_text(text):
    """Cleans excessive newlines and splits text into paragraphs or chunks."""
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', text.strip())  # Remove excessive newlines and spaces
    return cleaned_text.split("\n\n")  # Split based on double newlines (paragraphs)

def initialize_chromadb_collection(epub_path):
    """Initializes ChromaDB and returns the collection."""
    vectorstore = chromadb.PersistentClient(path=DB_PATH)
    collection_name = generate_valid_collection_name(epub_path)
    collection = vectorstore.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")
    return collection

def index_documents(collection, text_chunks):
    """Indexes the documents (text chunks) into ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    for idx, chunk in enumerate(text_chunks):
        doc_id = f"doc_{idx}"
        print(f"Indexing document {idx}: {chunk[:100]}")  # Print first 100 characters of each chunk
        collection.add(
            ids=[doc_id],
            documents=[chunk],
            embeddings=[embeddings.embed_query(chunk)]
        )

    # Verify documents are indexed
    documents_in_collection = collection.count()
    print(f"Number of documents indexed: {documents_in_collection}")

def save_to_db(epub_path):
    text_content, title = extract_text_from_epub(epub_path)
    text_chunks = clean_and_split_text(text_content)

    if not text_chunks:
        raise ValueError("No meaningful content to index.")

    collection = initialize_chromadb_collection(epub_path)
    index_documents(collection, text_chunks)

    # List all collections to verify
    collections = collection._client.list_collections()
    print(f"All collections: {[col.name for col in collections]}")
    print(f"EPUB content titled '{title}' has been indexed.")

if __name__ == "__main__":
    save_to_db(EPUB_FILE_PATH)
