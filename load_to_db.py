import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import re


# TODO Fix warnings
# TODO Save collection name and book title to file

# Generate a valid collection name based on the file name by sanitizing input
def generate_valid_collection_name(epub_path):
    file_name = os.path.basename(epub_path)
    base_name, _ = os.path.splitext(file_name)

    # Replace invalid characters with underscores and limit to 63 characters
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
            text_content += item.get_content().decode('utf-8')
    return text_content, title


# Load the EPUB content and save it to ChromaDB
def save_to_db(epub_path):
    epub_text, title = load_epub(epub_path)
    # print(f"Parsed EPUB text (first 500 characters): {epub_text[:500]}")  # To check a portion of the content

    # Write the text content to a temporary file with UTF-8 encoding
    with open("temp_epub_text.txt", "w", encoding="utf-8") as f:
        f.write(epub_text)

    # Load the content for ChromaDB indexing
    loader = TextLoader("temp_epub_text.txt", encoding="utf-8")

    # Use HuggingFaceEmbeddings for embedding the document
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize the ChromaDB client and create a valid collection name
    vectorstore = chromadb.PersistentClient(path='db')
    collection_name = generate_valid_collection_name(epub_path)

    # Create a collection with the sanitized name in ChromaDB
    collection = vectorstore.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    # Index the documents with embeddings in the ChromaDB collection
    index_creator = VectorstoreIndexCreator(embedding=embeddings)
    index_creator.from_documents(loader.load())

    # Check how many documents were indexed after indexing
    documents_in_collection = vectorstore.get_collection(collection_name).count()
    print(f"Number of documents indexed: {documents_in_collection}")

    # List all collections to verify
    collections = vectorstore.list_collections()
    print(f"All collections: {[col.name for col in collections]}")

    # Confirm success
    print(f"EPUB content titled '{title}' has been indexed with collection name: '{collection_name}'.")


if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'
    save_to_db(epub_path)
