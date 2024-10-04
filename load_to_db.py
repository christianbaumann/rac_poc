import ebooklib
from ebooklib import epub
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from chromadb import Chroma


# Load EPUB file
def load_epub(file_path):
    book = epub.read_epub(file_path)
    text_content = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text_content += item.get_content().decode('utf-8')
    return text_content


# Save to ChromaDB
def save_to_db(epub_path):
    epub_text = load_epub(epub_path)

    # Create a document loader
    loader = TextLoader(epub_text)

    # Set up ChromaDB
    vectorstore = Chroma(persist_directory='db')  # Specify persistent storage
    VectorstoreIndexCreator(vectorstore).create_from_loader(loader)

    print("EPUB content has been loaded and indexed in the database.")


if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'  # Adjust to your ebook location
    save_to_db(epub_path)
