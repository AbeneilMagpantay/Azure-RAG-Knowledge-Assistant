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
    """Complete RAG chain combining retrieval and LLM generation."""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        llm_provider: str = "azure",
        model: str = "gpt-4",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client."""
        if self.llm_provider == "azure":
            from openai import AzureOpenAI
            if not self.azure_endpoint or not self.azure_api_key:
                raise ValueError("Azure endpoint and API key required")
            self._client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version
            )
        elif self.llm_provider == "openai":
            from openai import OpenAI
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required")
            self._client = OpenAI(api_key=self.openai_api_key)
        elif self.llm_provider == "google":
            from google import genai
            if not self.google_api_key:
                raise ValueError("Google API key required")
            self._client = genai.Client(api_key=self.google_api_key)
    
    def query(self, question: str, top_k: int = 5, filters: Optional[str] = None, use_multi_query: bool = False) -> RAGResponse:
        """Run the complete RAG pipeline."""
        if use_multi_query:
            retrieval_result = self.retriever.retrieve_with_multi_query(
                query=question, llm_client=self, num_queries=3, top_k=top_k, filters=filters
            )
        else:
            retrieval_result = self.retriever.retrieve(query=question, top_k=top_k, filters=filters)
        
        context = format_context(retrieval_result.results)
        prompt = format_qa_prompt(question, context)
        answer, tokens = self._generate(prompt)
        
        sources = [{
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "score": r.score,
            "content_preview": r.content[:200] + "..."
        } for r in retrieval_result.results]
        
        return RAGResponse(answer=answer, sources=sources, query=question, context_used=context, tokens_used=tokens)
    
    def query_stream(self, question: str, top_k: int = 5, filters: Optional[str] = None) -> Generator[str, None, RAGResponse]:
        """Stream RAG response."""
        retrieval_result = self.retriever.retrieve(query=question, top_k=top_k, filters=filters)
        context = format_context(retrieval_result.results)
        prompt = format_qa_prompt(question, context)
        
        full_response = ""
        for token in self._generate_stream(prompt):
            full_response += token
            yield token
        
        sources = [{"source": r.metadata.get("source"), "page": r.metadata.get("page"), "score": r.score} for r in retrieval_result.results]
        return RAGResponse(answer=full_response, sources=sources, query=question, context_used=context)
    
    def generate(self, prompt: str) -> str:
        """Generate text (used by multi-query retrieval)."""
        response, _ = self._generate(prompt)
        return response
    
    def _generate(self, prompt: str) -> tuple[str, int]:
        """Generate response using configured LLM."""
        if self.llm_provider in ("azure", "openai"):
            return self._generate_openai(prompt)
        elif self.llm_provider == "google":
            return self._generate_google(prompt)
        return self._generate_ollama(prompt)
    
    def _generate_openai(self, prompt: str) -> tuple[str, int]:
        """Generate using OpenAI/Azure OpenAI."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        model = self.azure_deployment if self.llm_provider == "azure" else self.model
        
        response = self._client.chat.completions.create(
            model=model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
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
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
            }
        )
        response.raise_for_status()
        return response.json()["response"], 0

    def _generate_google(self, prompt: str) -> tuple[str, int]:
        """Generate using Google Gemini."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config={"temperature": self.temperature, "max_output_tokens": self.max_tokens}
        )
        return response.text, 0
    
    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream generation."""
        if self.llm_provider in ("azure", "openai"):
            yield from self._stream_openai(prompt)
        elif self.llm_provider == "google":
            yield from self._stream_google(prompt)
        else:
            yield from self._stream_ollama(prompt)
    
    def _stream_openai(self, prompt: str) -> Generator[str, None, None]:
        """Stream using OpenAI/Azure OpenAI."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        model = self.azure_deployment if self.llm_provider == "azure" else self.model
        
        stream = self._client.chat.completions.create(
            model=model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _stream_ollama(self, prompt: str) -> Generator[str, None, None]:
        """Stream using Ollama."""
        import requests
        import json
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": True,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
            },
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]

    def _stream_google(self, prompt: str) -> Generator[str, None, None]:
        """Stream using Google Gemini."""
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config={"temperature": self.temperature, "max_output_tokens": self.max_tokens}
        ):
            if chunk.text:
                yield chunk.text
