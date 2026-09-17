"""
Enterprise RAG Pipeline
Uses: FastEmbed (lightweight, free, no GPU) + MongoDB Atlas Vector Search + ChatNVIDIA
"""
from typing import List
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from core.config import settings

# Lightweight, fast embedding model — no GPU, no API key needed
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    """Returns FastEmbed embedding model (runs locally on the server, very lightweight)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_chunk_file(file_path: str, filename: str) -> List[Document]:
    """Load a file and split into chunks."""
    documents = []
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith((".txt", ".md")):
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            raise Exception(f"Unsupported file type: {filename}. Use PDF, TXT, or MD.")
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise Exception(f"Document parsing error: {str(e)}")

    if not documents:
        raise Exception(f"No content could be extracted from {filename}.")

    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = filename
        doc.metadata["category"] = "knowledge_base"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {filename}")
    return chunks


def store_in_mongodb(chunks: List[Document]) -> bool:
    """
    Embed chunks using FastEmbed (free, local, lightweight)
    and store in MongoDB Atlas Vector Search.
    """
    if not settings.MONGODB_URI:
        raise Exception("ERROR: MONGODB_URI is not set in backend environment variables.")

    if not chunks:
        raise Exception("No chunks to store. The document might be empty.")

    try:
        # Free local embeddings — no API key required, very lightweight
        embeddings = get_embeddings()
        print(f"Using embedding model: {EMBEDDING_MODEL}")

        # MongoDB Atlas Vector Store
        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        # Store documents with embeddings
        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name="vector_index"
        )

        print(f"SUCCESS: Stored {len(chunks)} chunks in MongoDB Atlas")
        client.close()
        return True

    except Exception as e:
        print(f"ERROR storing in MongoDB: {e}")
        raise e


def process_single_file(file_path: str, filename: str):
    """Full pipeline: Load → Chunk → Embed → Store."""
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")
    print(f"{'='*50}")

    # Step 1: Load and chunk
    chunks = load_and_chunk_file(file_path, filename)

    # Step 2: Store in MongoDB with FastEmbed embeddings
    store_in_mongodb(chunks)

    # Step 3: Cleanup temp file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Cleaned up temp file: {file_path}")


def search_knowledge_base(query: str, top_k: int = 5) -> List[Document]:
    """
    Search the knowledge base using the same FastEmbed embeddings.
    Returns top-k relevant document chunks.
    """
    if not settings.MONGODB_URI:
        return []

    try:
        embeddings = get_embeddings()

        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )

        results = vector_store.similarity_search(query, k=top_k)
        print(f"Found {len(results)} relevant chunks for query: '{query}'")
        client.close()
        return results

    except Exception as e:
        print(f"ERROR searching knowledge base: {e}")
        return []
