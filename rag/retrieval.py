"""
Phase 6: Enterprise Retrieval Pipeline
Implements: Hybrid Search (Vector + BM25), Cross-Encoder Re-ranking, Context Compression
"""
from typing import List
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from pymongo import MongoClient
from core.config import settings
from rag.ingestion import get_embeddings

def get_enterprise_context(query: str, top_k: int = 3) -> str:
    """
    Executes the full Enterprise Retrieval Pipeline:
    1. Vector Search (Semantic)
    2. BM25 Search (Keyword)
    3. Ensemble (Hybrid Fusion)
    4. Re-ranking (Cross-Encoder)
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
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # 2. BM25 Retriever (In-Memory Fallback representation for Hybrid)
        # Note: In a true massive enterprise, we use MongoDB Atlas Full-Text Search.
        # For this implementation, we pull recent docs to build BM25 dynamically,
        # or rely on Vector Search heavily. To keep it fast, we fetch raw docs.
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
        bm25_retriever.k = 10

        # 3. Hybrid Ensemble Retriever (Reciprocal Rank Fusion)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

        # 4. Context Compression & Re-ranking (Flashrank is ultra-lightweight ONNX)
        compressor = FlashrankRerank(top_n=top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )

        # Execute Pipeline
        compressed_docs = compression_retriever.invoke(query)
        
        if not compressed_docs:
            return ""

        # Format Context with Strict Citations
        context_parts = []
        for i, d in enumerate(compressed_docs, 1):
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
