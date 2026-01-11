"""RAG retriever with query enhancement."""

from typing import List, Optional, Union
from dataclasses import dataclass

from ..embeddings import EmbeddingGenerator
from ..vectorstore.azure_search import AzureSearchVectorStore, SearchResult
from ..vectorstore.local_store import ChromaVectorStore


@dataclass
class RetrievalResult:
    """Result from retrieval with context."""
    
    results: List[SearchResult]
    query: str
    enhanced_queries: List[str] = None


class RAGRetriever:
    """
    Retriever for RAG pipeline with query enhancement.
    
    Features:
    - Vector similarity search
    - Multi-query retrieval for better recall
    - Reranking (optional)
    - Hybrid search support
    """
    
    def __init__(
        self,
        vector_store: Union[AzureSearchVectorStore, ChromaVectorStore],
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5,
        search_type: str = "hybrid"
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            top_k: Number of results to retrieve
            search_type: "vector", "keyword", or "hybrid"
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.search_type = search_type
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Override default top_k
            filters: Optional filter expression
            
        Returns:
            RetrievalResult with search results
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding.embedding,
            query_text=query,
            top_k=k,
            filters=filters,
            search_type=self.search_type
        )
        
        return RetrievalResult(
            results=results,
            query=query
        )
    
    def retrieve_with_multi_query(
        self,
        query: str,
        llm_client,
        num_queries: int = 3,
        top_k: Optional[int] = None,
        filters: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve using multiple query variations for better recall.
        
        Args:
            query: Original user query
            llm_client: LLM client for generating query variations
            num_queries: Number of query variations to generate
            top_k: Results per query
            filters: Optional filter expression
            
        Returns:
            RetrievalResult with deduplicated results
        """
        from .prompts import MULTI_QUERY_PROMPT
        
        k = top_k or self.top_k
        
        # Generate query variations
        prompt = MULTI_QUERY_PROMPT.format(
            question=query,
            num_queries=num_queries
        )
        
        response = llm_client.generate(prompt)
        enhanced_queries = [q.strip() for q in response.split("\n") if q.strip()]
        
        # Add original query
        all_queries = [query] + enhanced_queries[:num_queries]
        
        # Retrieve for each query
        all_results = []
        seen_chunks = set()
        
        for q in all_queries:
            query_embedding = self.embedding_generator.embed(q)
            
            results = self.vector_store.search(
                query_embedding=query_embedding.embedding,
                query_text=q,
                top_k=k,
                filters=filters,
                search_type=self.search_type
            )
            
            # Deduplicate by chunk_id
            for result in results:
                if result.chunk_id not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(result.chunk_id)
        
        # Sort by score and limit to top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        all_results = all_results[:k]
        
        return RetrievalResult(
            results=all_results,
            query=query,
            enhanced_queries=enhanced_queries
        )
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder (optional enhancement).
        
        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results after reranking
            
        Returns:
            Reranked results
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            # If cross-encoder not available, return original results
            return results[:top_k] if top_k else results
        
        k = top_k or self.top_k
        
        # Load cross-encoder model
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Prepare pairs for reranking
        pairs = [[query, r.content] for r in results]
        
        # Get reranking scores
        scores = model.predict(pairs)
        
        # Sort by reranking score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        reranked = []
        for result, score in scored_results[:k]:
            result.score = float(score)
            reranked.append(result)
        
        return reranked
