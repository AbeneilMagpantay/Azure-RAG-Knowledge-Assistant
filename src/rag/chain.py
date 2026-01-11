"""RAG chain combining retrieval and generation."""

import os
from typing import Optional, Generator, List, Dict, Any
from dataclasses import dataclass, field

from .retriever import RAGRetriever
from .prompts import SYSTEM_PROMPT, format_context, format_qa_prompt


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    context_used: str = ""
    tokens_used: int = 0


class RAGChain:
    """
    Complete RAG chain combining retrieval and generation.
    
    Supports:
    - Azure OpenAI GPT-4
    - OpenAI API
    - Ollama (local)
    - Streaming responses
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        llm_provider: str = "azure",
        model: str = "gpt-4",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: RAGRetriever instance
            llm_provider: "azure", "openai", or "ollama"
            model: Model name
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            openai_api_key: OpenAI API key
            ollama_base_url: Ollama server URL
            ollama_model: Ollama model name
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Azure config
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        # OpenAI config
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Ollama config
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client."""
        if self.llm_provider == "azure":
            self._init_azure_client()
        elif self.llm_provider == "openai":
            self._init_openai_client()
        elif self.llm_provider == "ollama":
            # Ollama uses HTTP requests, no client needed
            pass
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
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
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[str] = None,
        use_multi_query: bool = False
    ) -> RAGResponse:
        """
        Run the complete RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filters: Optional filter expression
            use_multi_query: Use multi-query retrieval
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant documents
        if use_multi_query:
            retrieval_result = self.retriever.retrieve_with_multi_query(
                query=question,
                llm_client=self,  # Use self as LLM client
                num_queries=3,
                top_k=top_k,
                filters=filters
            )
        else:
            retrieval_result = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                filters=filters
            )
        
        # Format context
        context = format_context(retrieval_result.results)
        
        # Generate answer
        prompt = format_qa_prompt(question, context)
        answer, tokens = self._generate(prompt)
        
        # Extract sources
        sources = []
        for result in retrieval_result.results:
            sources.append({
                "source": result.metadata.get("source"),
                "page": result.metadata.get("page"),
                "score": result.score,
                "content_preview": result.content[:200] + "..."
            })
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
            tokens_used=tokens
        )
    
    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[str] = None
    ) -> Generator[str, None, RAGResponse]:
        """
        Stream RAG response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filters: Optional filter expression
            
        Yields:
            Response tokens as they're generated
            
        Returns:
            Final RAGResponse
        """
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filters=filters
        )
        
        # Format context
        context = format_context(retrieval_result.results)
        prompt = format_qa_prompt(question, context)
        
        # Stream generation
        full_response = ""
        for token in self._generate_stream(prompt):
            full_response += token
            yield token
        
        # Extract sources
        sources = []
        for result in retrieval_result.results:
            sources.append({
                "source": result.metadata.get("source"),
                "page": result.metadata.get("page"),
                "score": result.score
            })
        
        return RAGResponse(
            answer=full_response,
            sources=sources,
            query=question,
            context_used=context
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text (used by multi-query retrieval).
        
        Args:
            prompt: Prompt text
            
        Returns:
            Generated text
        """
        response, _ = self._generate(prompt)
        return response
    
    def _generate(self, prompt: str) -> tuple[str, int]:
        """Generate response using configured LLM."""
        if self.llm_provider in ("azure", "openai"):
            return self._generate_openai(prompt)
        else:
            return self._generate_ollama(prompt)
    
    def _generate_openai(self, prompt: str) -> tuple[str, int]:
        """Generate using OpenAI/Azure OpenAI."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        if self.llm_provider == "azure":
            response = self._client.chat.completions.create(
                model=self.azure_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        return response.choices[0].message.content, response.usage.total_tokens
    
    def _generate_ollama(self, prompt: str) -> tuple[str, int]:
        """Generate using Ollama."""
        import requests
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
        )
        response.raise_for_status()
        
        result = response.json()
        return result["response"], 0  # Ollama doesn't report tokens
    
    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream generation."""
        if self.llm_provider in ("azure", "openai"):
            yield from self._stream_openai(prompt)
        else:
            yield from self._stream_ollama(prompt)
    
    def _stream_openai(self, prompt: str) -> Generator[str, None, None]:
        """Stream using OpenAI/Azure OpenAI."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        if self.llm_provider == "azure":
            stream = self._client.chat.completions.create(
                model=self.azure_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
        else:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _stream_ollama(self, prompt: str) -> Generator[str, None, None]:
        """Stream using Ollama."""
        import requests
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
