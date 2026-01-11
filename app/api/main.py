"""
Azure RAG Knowledge Assistant - FastAPI REST API

Production-ready API for the RAG Knowledge Assistant.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    top_k: int = 5
    use_multi_query: bool = False
    filters: Optional[str] = None


class SourceInfo(BaseModel):
    """Source information in response."""
    source: str
    page: Optional[int] = None
    score: float
    content_preview: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[SourceInfo]
    query: str
    tokens_used: int = 0


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    texts: List[str]
    metadatas: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    rag_initialized: bool


# Global state
app_state = {
    "rag_chain": None,
    "vector_store": None,
    "embedding_gen": None,
    "initialized": False
}


def initialize_rag():
    """Initialize RAG components."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        # Check for provider preference
        use_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
        use_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        # Initialize embedding generator
        if use_azure:
            embedding_gen = EmbeddingGenerator(provider="azure")
        elif use_openai:
            embedding_gen = EmbeddingGenerator(provider="openai")
        else:
            embedding_gen = EmbeddingGenerator(provider="local")
        
        # Initialize vector store (using local for simplicity)
        vector_store = ChromaVectorStore()
        vector_store.create_index()
        
        # Initialize retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_gen
        )
        
        # Initialize RAG chain
        if use_azure:
            rag_chain = RAGChain(retriever=retriever, llm_provider="azure")
        elif use_openai:
            rag_chain = RAGChain(retriever=retriever, llm_provider="openai")
        else:
            rag_chain = RAGChain(retriever=retriever, llm_provider="ollama")
        
        app_state["rag_chain"] = rag_chain
        app_state["vector_store"] = vector_store
        app_state["embedding_gen"] = embedding_gen
        app_state["initialized"] = True
        
        print("✅ RAG initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        app_state["initialized"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    initialize_rag()
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="Azure RAG Knowledge Assistant API",
    description="REST API for the RAG Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        rag_initialized=app_state["initialized"]
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Send a question and receive an answer with sources.
    """
    if not app_state["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        response = app_state["rag_chain"].query(
            question=request.question,
            top_k=request.top_k,
            use_multi_query=request.use_multi_query,
            filters=request.filters
        )
        
        sources = [
            SourceInfo(
                source=s.get("source", "unknown"),
                page=s.get("page"),
                score=s.get("score", 0.0),
                content_preview=s.get("content_preview")
            )
            for s in response.sources
        ]
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            query=response.query,
            tokens_used=response.tokens_used
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/ingest")
async def ingest_texts(request: IngestRequest):
    """
    Ingest text documents into the knowledge base.
    
    Provide a list of texts to add to the vector store.
    """
    if not app_state["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        from src.document_processor.chunker import Chunk
        
        # Create chunks from texts
        chunks = []
        for i, text in enumerate(request.texts):
            metadata = request.metadatas[i] if request.metadatas else {}
            chunks.append(Chunk(
                content=text,
                metadata=metadata,
                chunk_id=f"api_chunk_{i}"
            ))
        
        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = app_state["embedding_gen"].embed_batch(texts)
        embedding_vectors = [e.embedding for e in embeddings]
        
        # Add to vector store
        app_state["vector_store"].add_documents(chunks, embedding_vectors)
        
        return {"status": "success", "chunks_added": len(chunks)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file into the knowledge base.
    
    Supported formats: PDF, TXT, DOCX, MD
    """
    if not app_state["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        from src.document_processor import DocumentLoader, DocumentChunker
        
        # Save temporarily
        temp_path = Path(f"./temp_{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)
        
        try:
            # Load and chunk
            loader = DocumentLoader()
            chunker = DocumentChunker()
            
            docs = loader.load(str(temp_path))
            chunks = chunker.chunk(docs)
            
            # Generate embeddings
            texts = [c.content for c in chunks]
            embeddings = app_state["embedding_gen"].embed_batch(texts)
            embedding_vectors = [e.embedding for e in embeddings]
            
            # Add to vector store
            app_state["vector_store"].add_documents(chunks, embedding_vectors)
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_added": len(chunks)
            }
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File ingestion failed: {str(e)}"
        )


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the knowledge base."""
    if not app_state["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        app_state["vector_store"].delete_index()
        app_state["vector_store"].create_index()
        return {"status": "success", "message": "All documents cleared"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Clear failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
