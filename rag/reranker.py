import os
from typing import List
from langchain_core.documents import Document
from core.config import settings

def rerank_documents(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """
    Reranks documents using Cohere's Cross-Encoder model.
    This provides highly accurate semantic ranking of the retrieved candidates.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        print("COHERE_API_KEY not found, skipping reranking.")
        return documents[:top_n]
        
    try:
        import cohere
        co = cohere.Client(cohere_api_key)
        
        docs_text = [doc.page_content for doc in documents]
        results = co.rerank(
            query=query,
            documents=docs_text,
            top_n=top_n,
            model="rerank-english-v3.0"
        )
        
        reranked_docs = []
        for res in results.results:
            original_doc = documents[res.index]
            original_doc.metadata["relevance_score"] = res.relevance_score
            reranked_docs.append(original_doc)
            
        return reranked_docs
    except Exception as e:
        print(f"Reranking failed: {e}")
        return documents[:top_n]
