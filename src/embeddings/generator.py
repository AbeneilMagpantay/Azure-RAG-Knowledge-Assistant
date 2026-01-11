"""Embedding generation using Azure OpenAI or local models."""

import os
from typing import List, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class EmbeddingResult:
    """Embedding generation result."""
    embedding: List[float]
    text: str
    model: str
    tokens_used: int = 0


class EmbeddingGenerator:
    """Generate embeddings using Azure OpenAI, OpenAI API, or local sentence-transformers."""
    
    DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "models/embedding-001": 768,
        "models/text-embedding-004": 768,
    }
    
    def __init__(
        self,
        provider: str = "azure",
        model: str = "text-embedding-3-large",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        local_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.provider = provider
        self.model = model
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.local_model_name = local_model_name
        
        self._client = None
        self._local_model = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "azure":
            from openai import AzureOpenAI
            if not self.azure_endpoint or not self.azure_api_key:
                raise ValueError("Azure endpoint and API key required")
            self._client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version
            )
        elif self.provider == "openai":
            from openai import OpenAI
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required")
            self._client = OpenAI(api_key=self.openai_api_key)
        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.local_model_name)
        elif self.provider == "google":
            import google.generativeai as genai
            if not self.google_api_key:
                raise ValueError("Google API key required")
            genai.configure(api_key=self.google_api_key)
            self._client = genai
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if self.provider == "local":
            embedding = self._local_model.encode(text).tolist()
            return EmbeddingResult(embedding=embedding, text=text, model=self.local_model_name)
        
        if self.provider == "google":
            result = self._client.embed_content(
                model=self.model,
                content=text
            )
            return EmbeddingResult(embedding=result['embedding'], text=text, model=self.model)
        
        model = self.azure_deployment if self.provider == "azure" else self.model
        response = self._client.embeddings.create(input=text, model=model)
        
        return EmbeddingResult(
            embedding=response.data[0].embedding,
            text=text,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )
    
    def embed_batch(self, texts: List[str], batch_size: int = 100, show_progress: bool = True) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        results = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings") if show_progress else range(0, len(texts), batch_size)
        except ImportError:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            if self.provider == "local":
                embeddings = self._local_model.encode(batch)
                for j, embedding in enumerate(embeddings):
                    results.append(EmbeddingResult(embedding=embedding.tolist(), text=batch[j], model=self.local_model_name))
            elif self.provider == "google":
                # batch embedding for google
                result = self._client.embed_content(
                    model=self.model,
                    content=batch
                )
                # check if result['embedding'] is list of lists
                embeddings = result['embedding']
                for j, emb in enumerate(embeddings):
                    results.append(EmbeddingResult(embedding=emb, text=batch[j], model=self.model))
            else:
                model = self.azure_deployment if self.provider == "azure" else self.model
                response = self._client.embeddings.create(input=batch, model=model)
                
                for j, data in enumerate(response.data):
                    results.append(EmbeddingResult(
                        embedding=data.embedding,
                        text=batch[j],
                        model=self.model,
                        tokens_used=response.usage.total_tokens // len(batch)
                    ))
        
        return results
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the current model."""
        model_key = self.local_model_name if self.provider == "local" else self.model
        return self.DIMENSIONS.get(model_key, 1536)
