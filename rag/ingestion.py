"""
Phase 6: Enterprise RAG Ingestion Pipeline
Supports: PDF, DOCX, MD, HTML, CSV
Enhances Metadata: version, effective_date, access_level, etc.
"""
from typing import List
import os
import datetime
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    CSVLoader,
    BSHTMLLoader
)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from core.config import settings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_embeddings():
    """Returns FastEmbed embedding model (local, lightweight)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

def load_and_chunk_file(file_path: str, filename: str) -> List[Document]:
    """Parse various file formats and chunk them with enriched metadata."""
    documents = []
    try:
        if file_path.endswith(".pdf"):
            documents = PyPDFLoader(file_path).load()
        elif file_path.endswith((".txt", ".md")):
            documents = TextLoader(file_path).load()
        elif file_path.endswith(".docx"):
            documents = Docx2txtLoader(file_path).load()
        elif file_path.endswith(".csv"):
            documents = CSVLoader(file_path).load()
        elif file_path.endswith((".html", ".htm")):
            documents = BSHTMLLoader(file_path).load()
        else:
            raise Exception(f"Unsupported file format: {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise Exception(f"Document parsing error: {str(e)}")

    if not documents:
        raise Exception(f"No content extracted from {filename}.")

    # Metadata Extraction Pipeline
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    for doc in documents:
        doc.metadata["document_id"] = str(uuid.uuid4())
        doc.metadata["source"] = filename
        doc.metadata["version"] = "1.0"
        doc.metadata["effective_date"] = current_date
        doc.metadata["access_level"] = "public"
        
        # Simple categorization heuristic
        filename_lower = filename.lower()
        if "policy" in filename_lower: doc.metadata["category"] = "policy"
        elif "faq" in filename_lower: doc.metadata["category"] = "faq"
        elif "manual" in filename_lower: doc.metadata["category"] = "manual"
        elif "price" in filename_lower: doc.metadata["category"] = "pricing"
        else: doc.metadata["category"] = "general"

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {filename}")
    return chunks

def store_in_mongodb(chunks: List[Document]) -> bool:
    """Embed and store chunks in MongoDB Atlas Vector Search."""
    if not settings.MONGODB_URI:
        raise Exception("MONGODB_URI not set.")
    if not chunks:
        raise Exception("No chunks to store.")

    try:
        embeddings = get_embeddings()
        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        # Enterprise storage with vector embeddings
        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name="vector_index"
        )
        print(f"SUCCESS: Stored {len(chunks)} chunks in MongoDB.")
        client.close()
        return True
    except Exception as e:
        print(f"ERROR storing in MongoDB: {e}")
        raise e

def process_single_file(file_path: str, filename: str):
    """Pipeline: Parsing -> Metadata -> Chunking -> Embedding -> Vector Store."""
    print(f"Enterprise Processing: {filename}")
    chunks = load_and_chunk_file(file_path, filename)
    store_in_mongodb(chunks)
    
    if os.path.exists(file_path):
        os.remove(file_path)
