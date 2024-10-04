import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import os
import re


# TODO Save collection name and book title to file

def generate_valid_collection_name(epub_path):
    file_name = os.path.basename(epub_path)
    base_name, _ = os.path.splitext(file_name)
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    return valid_name[:63]


def load_epub(file_path):
    book = epub.read_epub(file_path)
    text_content = ''
    title = book.get_metadata('DC', 'title')[0][0]

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text_content += item.get_content().decode('utf-8')
    return text_content, title


def save_to_db(epub_path):
    epub_text, title = load_epub(epub_path)

    with open("temp_epub_text.txt", "w", encoding="utf-8") as f:
        f.write(epub_text)

    loader = TextLoader("temp_epub_text.txt", encoding="utf-8")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = chromadb.PersistentClient(path='db')
    collection_name = generate_valid_collection_name(epub_path)

    collection = vectorstore.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    index_creator = VectorstoreIndexCreator(vectorstore=vectorstore, embedding=embeddings)
    index_creator.from_documents(loader.load())

    documents_in_collection = vectorstore.get_collection(collection_name).count()
    print(f"Number of documents indexed: {documents_in_collection}")

    collections = vectorstore.list_collections()
    print(f"All collections: {[col.name for col in collections]}")

    print(f"EPUB content titled '{title}' has been indexed with collection name: '{collection_name}'.")


if __name__ == "__main__":
    epub_path = 'ebooks/howcanitestthis.epub'
    save_to_db(epub_path)
