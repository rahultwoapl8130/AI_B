"""
Phase 6: Enterprise Retrieval Pipeline
Implements: Hybrid Search (Vector + BM25)
Note: Local Re-ranking (Flashrank) removed to prevent Render Free Tier Out-Of-Memory (OOM) crashes.
"""
from typing import List
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pymongo import MongoClient
from core.config import settings
from rag.ingestion import get_embeddings

def get_enterprise_context(query: str, top_k: int = 3) -> str:
    """
    Executes the Enterprise Retrieval Pipeline:
    Simplified to Vector Search only to prevent Render Free Tier Timeouts & OOMs.
    """
    if not settings.MONGODB_URI:
        return ""

    client = None
    try:
        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        # Vector Retriever Only (Fast & Memory Efficient)
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=get_embeddings(),
            index_name="vector_index"
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        # Retrieve docs
        final_docs = vector_retriever.invoke(query)
        
        if not final_docs:
            return ""

        # Format Context with Strict Citations
        context_parts = []
        for i, d in enumerate(final_docs, 1):
            source = d.metadata.get("source", "Unknown")
            category = d.metadata.get("category", "General")
            ver = d.metadata.get("version", "1.0")
            context_parts.append(f"--- [CITATION: {source} | Category: {category} | v{ver}] ---\n{d.page_content}")
            
        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"Enterprise Retrieval Error: {e}")
        return ""
    finally:
        if client:
            client.close()
