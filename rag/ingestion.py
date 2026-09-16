from typing import List
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from core.config import settings

# This script would typically be run as a background job or CLI script 
# to index documents into the Vector Store.

def load_documents(directory_path: str) -> List[Document]:
    """Loads PDFs and Text documents from a directory."""
    documents = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def clean_and_chunk(documents: List[Document]) -> List[Document]:
    """Cleans text and chunks documents for vectorization."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add rich metadata for filtering
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": f"chunk_{i}",
            "source": chunk.metadata.get("source", "unknown"),
            "category": "support_kb",
            "access_level": "public"
        })
    return chunks

def ingest_to_vector_store(chunks: List[Document]):
    """Embeds chunks and stores them in MongoDB Atlas."""
    if not settings.MONGODB_URI or not settings.OPENAI_API_KEY:
        print("Missing MONGODB_URI or OPENAI_API_KEY. Skipping ingestion.")
        return
        
    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.PROJECT_NAME.replace(" ", "_")]
    collection = db["vector_knowledge_base"]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create or update the vector store
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name="vector_index"
    )
    print(f"Successfully ingested {len(chunks)} chunks into MongoDB Atlas.")

def process_single_file(file_path: str, filename: str):
    """Processes a single file and ingests it."""
    print(f"Processing uploaded file: {filename}")
    documents = []
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
        
    for doc in documents:
        doc.metadata["source"] = filename
        
    chunks = clean_and_chunk(documents)
    ingest_to_vector_store(chunks)
    
    # Clean up temp file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted temp file: {file_path}")

if __name__ == "__main__":
    # Example usage:
    # docs = load_documents("./data")
    # chunks = clean_and_chunk(docs)
    # ingest_to_vector_store(chunks)
    pass
