import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb


# Load EPUB file and extract title
def load_epub(file_path):
    book = epub.read_epub(file_path)
    text_content = ''
    title = book.get_metadata('DC', 'title')[0][0]  # Extract the title from the EPUB metadata
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text_content += item.get_content().decode('utf-8')
    return text_content, title


# Save to ChromaDB
def save_to_db(epub_path):
    epub_text, title = load_epub(epub_path)  # Get EPUB content and title

    # Write EPUB text to temporary file
    with open("temp_epub_text.txt", "w", encoding="utf-8") as f:  # Set encoding to UTF-8 to handle special characters
        f.write(epub_text)

    loader = TextLoader("temp_epub_text.txt",
                        encoding="utf-8")  # Ensure the loader reads the file with the correct encoding

    # Use HuggingFaceEmbeddings from the updated package
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Set up ChromaDB client
    vectorstore = chromadb.PersistentClient(path='db')

    # Create a collection with the EPUB title as the collection name
    collection = vectorstore.create_collection(name=title)

    # Create index with embeddings using `from_documents` method
    index_creator = VectorstoreIndexCreator(embedding=embeddings)
    index_creator.from_documents(loader.load())

    print(f"EPUB content titled '{title}' has been loaded and indexed in the database.")


if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'
    save_to_db(epub_path)
