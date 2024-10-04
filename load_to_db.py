import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import chromadb


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

    # Use TextLoader directly with a file path instead of text argument
    with open("temp_epub_text.txt", "w", encoding="utf-8") as f:
        f.write(epub_text)

    loader = TextLoader("temp_epub_text.txt")  # TextLoader requires a file path

    # Set up ChromaDB
    vectorstore = chromadb.PersistentClient(path='db')
    index_creator = VectorstoreIndexCreator(vectorstore)
    index_creator.create_from_loader(loader)  # Fixed method

    print("EPUB content has been loaded and indexed in the database.")




if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'
    save_to_db(epub_path)
