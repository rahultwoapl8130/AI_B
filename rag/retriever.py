from typing import List, Dict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from rag.reranker import rerank_documents

def reciprocal_rank_fusion(vector_results: List[Document], bm25_results: List[Document], k=60) -> List[Document]:
    """
    Fuses rankings from multiple retrievers using Reciprocal Rank Fusion (RRF).
    RRF score = 1 / (k + rank)
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # Process Vector Results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        doc_map[doc_id] = doc
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0.0
        fused_scores[doc_id] += 1 / (rank + k)
        
    # Process BM25 Results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
        doc_map[doc_id] = doc
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0.0
        fused_scores[doc_id] += 1 / (rank + k)
        
    # Sort documents by their RRF score
    reranked_results = [
        (score, doc_map[doc_id]) for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Return sorted documents
    return [doc for score, doc in reranked_results]

def mock_vector_search(query: str) -> List[Document]:
    # Placeholder for MongoDB Atlas Vector Search
    return [Document(page_content="Vector Search: Refund policy says 30 days.", metadata={"chunk_id": "v1"})]

def mock_bm25_search(query: str) -> List[Document]:
    # Placeholder for BM25 Lexical Search
    return [Document(page_content="BM25 Keyword: Error code ERR-509 indicates a timeout.", metadata={"chunk_id": "b1"})]

def hybrid_search(query: str, top_k: int = 5) -> List[Document]:
    """
    Executes the Hybrid RAG pipeline: 
    1. Vector Search (Semantic)
    2. BM25 Search (Keyword)
    3. RRF Fusion
    4. Cross-Encoder Reranking
    """
    # 1 & 2: Parallel Retrieval
    vector_results = mock_vector_search(query)
    bm25_results = mock_bm25_search(query)
    
    # 3: Fusion
    fused_documents = reciprocal_rank_fusion(vector_results, bm25_results)
    
    # 4: Reranking
    final_documents = rerank_documents(query, fused_documents, top_n=top_k)
    
    return final_documents
