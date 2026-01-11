"""Azure AI Search vector store implementation."""

import os
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from vector search."""
    
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str


class AzureSearchVectorStore:
    """
    Vector store using Azure AI Search.
    
    Features:
    - Vector similarity search
    - Hybrid search (vector + keyword)
    - Metadata filtering
    - Automatic index management
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "rag-knowledge-index",
        embedding_dimension: int = 3072
    ):
        """
        Initialize Azure AI Search vector store.
        
        Args:
            endpoint: Azure AI Search endpoint
            api_key: Azure AI Search admin key
            index_name: Name of the search index
            embedding_dimension: Dimension of embedding vectors
        """
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Search endpoint and API key are required")
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Azure Search clients."""
        try:
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-search-documents required. "
                "Install with: pip install azure-search-documents"
            )
        
        credential = AzureKeyCredential(self.api_key)
        
        self._index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=credential
        )
        
        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=credential
        )
    
    def create_index(self, recreate: bool = False):
        """
        Create the search index with vector configuration.
        
        Args:
            recreate: If True, delete existing index first
        """
        from azure.search.documents.indexes.models import (
            SearchIndex,
            SearchField,
            SearchFieldDataType,
            SimpleField,
            SearchableField,
            VectorSearch,
            HnswAlgorithmConfiguration,
            VectorSearchProfile,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
        )
        
        # Delete existing index if recreate is True
        if recreate:
            try:
                self._index_client.delete_index(self.index_name)
                print(f"Deleted existing index: {self.index_name}")
            except Exception:
                pass
        
        # Define fields
        fields = [
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dimension,
                vector_search_profile_name="vector-profile"
            ),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="page",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            SearchableField(
                name="file_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-config")
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        self._index_client.create_or_update_index(index)
        print(f"Created index: {self.index_name}")
    
    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize a document key for Azure AI Search.
        
        Azure AI Search only allows letters, digits, underscore (_), 
        dash (-), or equal sign (=) in document keys.
        """
        # Replace any character that's not allowed with underscore
        return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)
    
    def add_documents(
        self,
        chunks: List[Any],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """
        Add documents to the index.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            batch_size: Number of documents to upload at once
        """
        documents = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # Sanitize chunk_id to only contain allowed characters
            sanitized_id = self._sanitize_key(chunk.chunk_id)
            doc = {
                "chunk_id": sanitized_id,
                "content": chunk.content,
                "content_vector": embedding,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "file_type": chunk.metadata.get("file_type", "unknown")
            }
            documents.append(doc)
        
        # Upload in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._search_client.upload_documents(batch)
        
        print(f"Added {len(documents)} documents to index")
    
    def search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Optional text for hybrid search
            top_k: Number of results to return
            filters: OData filter expression
            search_type: "vector", "keyword", or "hybrid"
            
        Returns:
            List of SearchResult objects
        """
        from azure.search.documents.models import VectorizedQuery
        
        results = []
        
        if search_type == "vector":
            # Pure vector search
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            response = self._search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filters,
                top=top_k
            )
        elif search_type == "keyword":
            # Pure keyword search
            response = self._search_client.search(
                search_text=query_text,
                filter=filters,
                top=top_k
            )
        else:
            # Hybrid search (vector + keyword)
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            response = self._search_client.search(
                search_text=query_text or "",
                vector_queries=[vector_query],
                filter=filters,
                top=top_k
            )
        
        for result in response:
            results.append(SearchResult(
                content=result["content"],
                score=result["@search.score"],
                metadata={
                    "source": result.get("source"),
                    "page": result.get("page"),
                    "chunk_index": result.get("chunk_index"),
                    "file_type": result.get("file_type")
                },
                chunk_id=result["chunk_id"]
            ))
        
        return results
    
    def delete_documents(self, chunk_ids: List[str]):
        """Delete documents by chunk ID."""
        documents = [{"chunk_id": cid} for cid in chunk_ids]
        self._search_client.delete_documents(documents)
        print(f"Deleted {len(chunk_ids)} documents")
    
    def delete_index(self):
        """Delete the entire index."""
        self._index_client.delete_index(self.index_name)
        print(f"Deleted index: {self.index_name}")
