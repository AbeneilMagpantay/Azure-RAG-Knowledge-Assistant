"""Embedding generation using Azure OpenAI or local models."""

import os
from typing import List, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    embedding: List[float]
    text: str
    model: str
    tokens_used: int = 0


class EmbeddingGenerator:
    """
    Generate embeddings using Azure OpenAI or local sentence-transformers.
    
    Supports:
    - Azure OpenAI (text-embedding-3-large, text-embedding-ada-002)
    - OpenAI API (for development without Azure)
    - Sentence Transformers (fully local, free)
    """
    
    def __init__(
        self,
        provider: str = "azure",
        model: str = "text-embedding-3-large",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        local_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedding generator.
        
        Args:
            provider: "azure", "openai", or "local"
            model: Model name for Azure/OpenAI
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            openai_api_key: OpenAI API key (for non-Azure usage)
            local_model_name: Sentence Transformers model name
        """
        self.provider = provider
        self.model = model
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.local_model_name = local_model_name
        
        self._client = None
        self._local_model = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "azure":
            self._init_azure_client()
        elif self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "local":
            self._init_local_model()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        if not self.azure_endpoint or not self.azure_api_key:
            raise ValueError("Azure endpoint and API key are required")
        
        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version="2024-02-15-preview"
        )
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self._client = OpenAI(api_key=self.openai_api_key)
    
    def _init_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self._local_model = SentenceTransformer(self.local_model_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        if self.provider == "local":
            return self._embed_local(text)
        else:
            return self._embed_api(text)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            if self.provider == "local":
                batch_results = self._embed_local_batch(batch)
            else:
                batch_results = self._embed_api_batch(batch)
            
            results.extend(batch_results)
        
        return results
    
    def _embed_api(self, text: str) -> EmbeddingResult:
        """Embed single text using API."""
        if self.provider == "azure":
            response = self._client.embeddings.create(
                input=text,
                model=self.azure_deployment
            )
        else:
            response = self._client.embeddings.create(
                input=text,
                model=self.model
            )
        
        return EmbeddingResult(
            embedding=response.data[0].embedding,
            text=text,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )
    
    def _embed_api_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed batch of texts using API."""
        if self.provider == "azure":
            response = self._client.embeddings.create(
                input=texts,
                model=self.azure_deployment
            )
        else:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model
            )
        
        results = []
        for i, embedding_data in enumerate(response.data):
            results.append(EmbeddingResult(
                embedding=embedding_data.embedding,
                text=texts[i],
                model=self.model,
                tokens_used=response.usage.total_tokens // len(texts)
            ))
        
        return results
    
    def _embed_local(self, text: str) -> EmbeddingResult:
        """Embed single text using local model."""
        embedding = self._local_model.encode(text).tolist()
        
        return EmbeddingResult(
            embedding=embedding,
            text=text,
            model=self.local_model_name,
            tokens_used=0
        )
    
    def _embed_local_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed batch of texts using local model."""
        embeddings = self._local_model.encode(texts)
        
        results = []
        for i, embedding in enumerate(embeddings):
            results.append(EmbeddingResult(
                embedding=embedding.tolist(),
                text=texts[i],
                model=self.local_model_name,
                tokens_used=0
            ))
        
        return results
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        dimensions = {
            # Azure/OpenAI models
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
            # Local models
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }
        
        model_key = self.local_model_name if self.provider == "local" else self.model
        return dimensions.get(model_key, 1536)
