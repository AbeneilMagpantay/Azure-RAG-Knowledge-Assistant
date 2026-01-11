"""ChromaDB vector store for local development."""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from vector search."""
    
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str


class ChromaVectorStore:
    """
    Local vector store using ChromaDB.
    
    Perfect for:
    - Local development without Azure costs
    - Testing and prototyping
    - Offline usage
    
    Features same interface as AzureSearchVectorStore for easy swapping.
    """
    
    def __init__(
        self,
        collection_name: str = "rag-knowledge",
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 384  # Default for local models
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist data (None for in-memory)
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        self.embedding_dimension = embedding_dimension
        
        self._client = None
        self._collection = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb required. Install with: pip install chromadb"
            )
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
    
    def create_index(self, recreate: bool = False):
        """
        Create or get the collection.
        
        Args:
            recreate: If True, delete existing collection first
        """
        if recreate:
            try:
                self._client.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
        
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Created/loaded collection: {self.collection_name}")
    
    def add_documents(
        self,
        chunks: List[Any],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """
        Add documents to the collection.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            batch_size: Number of documents to upload at once
        """
        if self._collection is None:
            self.create_index()
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk, embedding in zip(chunks, embeddings):
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "source": str(chunk.metadata.get("source", "unknown")),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "file_type": str(chunk.metadata.get("file_type", "unknown"))
            })
        
        # Add in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_emb = embeddings[i:i + batch_size]
            
            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb
            )
        
        print(f"Added {len(ids)} documents to collection")
    
    def search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Optional text (not used in Chroma, kept for interface compatibility)
            top_k: Number of results to return
            filters: Chroma where filter
            search_type: Ignored (Chroma only does vector search)
            
        Returns:
            List of SearchResult objects
        """
        if self._collection is None:
            self.create_index()
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Chroma returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    chunk_id=chunk_id
                ))
        
        return search_results
    
    def delete_documents(self, chunk_ids: List[str]):
        """Delete documents by chunk ID."""
        if self._collection is None:
            return
        
        self._collection.delete(ids=chunk_ids)
        print(f"Deleted {len(chunk_ids)} documents")
    
    def delete_index(self):
        """Delete the entire collection."""
        try:
            self._client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception:
            pass
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()
