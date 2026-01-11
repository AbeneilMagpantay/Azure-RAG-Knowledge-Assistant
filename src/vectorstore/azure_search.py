"""Azure AI Search vector store implementation."""

import os
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result from vector store."""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str


class AzureSearchVectorStore:
    """Vector store using Azure AI Search with hybrid search support."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "rag-knowledge-index",
        embedding_dimension: int = 3072
    ):
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Search endpoint and API key required")
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Azure Search clients."""
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        
        credential = AzureKeyCredential(self.api_key)
        self._index_client = SearchIndexClient(endpoint=self.endpoint, credential=credential)
        self._search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=credential)
    
    def create_index(self, recreate: bool = False):
        """Create search index with vector and semantic configuration."""
        from azure.search.documents.indexes.models import (
            SearchIndex, SearchField, SearchFieldDataType, SimpleField,
            SearchableField, VectorSearch, HnswAlgorithmConfiguration,
            VectorSearchProfile, SemanticConfiguration, SemanticField,
            SemanticPrioritizedFields, SemanticSearch
        )
        
        if recreate:
            try:
                self._index_client.delete_index(self.index_name)
                print(f"Deleted existing index: {self.index_name}")
            except Exception:
                pass
        
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dimension,
                vector_search_profile_name="vector-profile"
            ),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True)
        ]
        
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")]
        )
        
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")])
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )
        
        self._index_client.create_or_update_index(index)
        print(f"Created index: {self.index_name}")
    
    def _sanitize_key(self, key: str) -> str:
        """Sanitize document key for Azure AI Search (alphanumeric, underscore, dash, equals only)."""
        return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)
    
    def add_documents(self, chunks: List[Any], embeddings: List[List[float]], batch_size: int = 100):
        """Add documents to the index."""
        documents = []
        
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "chunk_id": self._sanitize_key(chunk.chunk_id),
                "content": chunk.content,
                "content_vector": embedding,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "file_type": chunk.metadata.get("file_type", "unknown")
            })
        
        for i in range(0, len(documents), batch_size):
            self._search_client.upload_documents(documents[i:i + batch_size])
        
        print(f"Added {len(documents)} documents to index")
    
    def search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """Search for similar documents using vector, keyword, or hybrid search."""
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="content_vector")
        
        if search_type == "vector":
            response = self._search_client.search(search_text=None, vector_queries=[vector_query], filter=filters, top=top_k)
        elif search_type == "keyword":
            response = self._search_client.search(search_text=query_text, filter=filters, top=top_k)
        else:
            response = self._search_client.search(search_text=query_text or "", vector_queries=[vector_query], filter=filters, top=top_k)
        
        results = []
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
        self._search_client.delete_documents([{"chunk_id": cid} for cid in chunk_ids])
        print(f"Deleted {len(chunk_ids)} documents")
    
    def delete_index(self):
        """Delete the entire index."""
        self._index_client.delete_index(self.index_name)
        print(f"Deleted index: {self.index_name}")
