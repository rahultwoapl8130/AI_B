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
    1. Vector Search (Semantic)
    2. BM25 Search (Keyword)
    3. Ensemble (Hybrid Fusion via Reciprocal Rank Fusion)
    """
    if not settings.MONGODB_URI:
        return ""

    client = None
    try:
        client = MongoClient(settings.MONGODB_URI)
        db = client["techmart_db"]
        collection = db["vector_knowledge_base"]

        # 1. Vector Retriever
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=get_embeddings(),
            index_name="vector_index"
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 2. BM25 Retriever
        cursor = collection.find({}, {"text": 1, "source": 1, "category": 1, "version": 1}).limit(500)
        all_docs = []
        for doc in cursor:
            if "text" in doc:
                metadata = {
                    "source": doc.get("source", "Unknown"),
                    "category": doc.get("category", "general"),
                    "version": doc.get("version", "1.0")
                }
                all_docs.append(Document(page_content=doc["text"], metadata=metadata))
        
        if not all_docs:
            return ""
            
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5

        # 3. Hybrid Ensemble Retriever (Reciprocal Rank Fusion)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

        # Execute Pipeline (Retrieve top docs directly from ensemble, skipping heavy reranker)
        retrieved_docs = ensemble_retriever.invoke(query)
        
        # Limit to top_k to avoid overflowing LLM context
        final_docs = retrieved_docs[:top_k]
        
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
