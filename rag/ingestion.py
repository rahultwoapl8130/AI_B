"""
Enterprise RAG Pipeline — Full NVIDIA Stack
Uses: NVIDIAEmbeddings + MongoDB Atlas + NVIDIARerank + ChatNVIDIA
"""
from typing import List, Optional
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from core.config import settings


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
            print(f"Unsupported file type: {filename}")
            return []
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise Exception(f"Document parsing error: {str(e)}")

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
    Embed chunks using NVIDIA NV-Embed-QA model
    and store in MongoDB Atlas Vector Search.
    """
    if not settings.MONGODB_URI:
        raise Exception("ERROR: MONGODB_URI is not set in backend.")

    if not settings.NVIDIA_API_KEY:
        raise Exception("ERROR: NVIDIA_API_KEY is not set in backend.")

    if not chunks:
        raise Exception("No chunks to store. The document might be empty.")

    try:
        # NVIDIA Embeddings — NV-Embed-QA is optimized for RAG
        embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=settings.NVIDIA_API_KEY,
            truncate="END"
        )

        # MongoDB Atlas Vector Store
        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        # Store documents with embeddings
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name="vector_index"
        )

        print(f"SUCCESS: Stored {len(chunks)} chunks in MongoDB Atlas")
        return True

    except Exception as e:
        print(f"ERROR storing in MongoDB: {e}")
        raise e


def process_single_file(file_path: str, filename: str) -> bool:
    """Full pipeline: Load → Chunk → Embed → Store."""
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")
    print(f"{'='*50}")

    # Step 1: Load and chunk
    chunks = load_and_chunk_file(file_path, filename)
    if not chunks:
        raise Exception(f"Failed to extract text from {filename}. The file might be empty, corrupted, or unsupported.")

    # Step 2: Store in MongoDB with NVIDIA embeddings
    success = store_in_mongodb(chunks)

    # Step 3: Cleanup temp file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Cleaned up temp file: {file_path}")

    return success


def search_knowledge_base(query: str, top_k: int = 5) -> List[Document]:
    """
    Search the knowledge base using NVIDIA embeddings.
    Returns top-k relevant document chunks.
    """
    if not settings.MONGODB_URI or not settings.NVIDIA_API_KEY:
        return []

    try:
        embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=settings.NVIDIA_API_KEY,
            truncate="END"
        )

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
        return results

    except Exception as e:
        print(f"ERROR searching knowledge base: {e}")
        return []
