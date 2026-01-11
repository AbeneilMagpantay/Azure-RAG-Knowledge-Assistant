"""
Azure RAG Knowledge Assistant - Streamlit UI

A chat interface for the RAG Knowledge Assistant with:
- Document upload and ingestion
- Chat with citations
- Evaluation metrics display
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def setup_sidebar():
    """Setup sidebar with configuration options."""
    st.sidebar.title("Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "LLM Provider",
        ["azure", "openai", "ollama"],
        index=0
    )
    
    # Provider-specific settings
    if provider == "azure":
        st.sidebar.text_input(
            "Azure OpenAI Endpoint",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            key="azure_endpoint",
            type="password"
        )
        st.sidebar.text_input(
            "Azure OpenAI Key",
            value=os.getenv("AZURE_OPENAI_API_KEY", ""),
            key="azure_key",
            type="password"
        )
        st.sidebar.text_input(
            "Deployment Name",
            value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            key="azure_deployment"
        )
    elif provider == "openai":
        st.sidebar.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="openai_key",
            type="password"
        )
    else:  # ollama
        st.sidebar.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            key="ollama_url"
        )
        st.sidebar.text_input(
            "Ollama Model",
            value="llama3",
            key="ollama_model"
        )
    
    # Search settings
    st.sidebar.subheader("Search")
    top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["hybrid", "vector", "keyword"]
    )
    
    # Vector store selection
    st.sidebar.subheader("Storage")
    use_azure_search = st.sidebar.checkbox(
        "Use Azure AI Search",
        value=bool(os.getenv("AZURE_SEARCH_ENDPOINT"))
    )
    
    return {
        "provider": provider,
        "top_k": top_k,
        "search_type": search_type,
        "use_azure_search": use_azure_search
    }


def init_rag_chain(config):
    """Initialize RAG chain with current configuration."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.vectorstore.azure_search import AzureSearchVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        # Initialize embedding generator
        # IMPORTANT: Always use Azure embeddings when using Azure Search!
        if config["use_azure_search"]:
            # Use Azure OpenAI embeddings for Azure Search (1536 dims for ada-002)
            embedding_gen = EmbeddingGenerator(provider="azure")
        elif config["provider"] == "azure":
            embedding_gen = EmbeddingGenerator(
                provider="azure",
                azure_endpoint=st.session_state.get("azure_endpoint"),
                azure_api_key=st.session_state.get("azure_key")
            )
        elif config["provider"] == "openai":
            embedding_gen = EmbeddingGenerator(
                provider="openai",
                openai_api_key=st.session_state.get("openai_key")
            )
        else:
            embedding_gen = EmbeddingGenerator(provider="local")
        
        # Initialize vector store
        # Use 3072 dimensions for text-embedding-3-large
        if config["use_azure_search"]:
            vector_store = AzureSearchVectorStore(embedding_dimension=3072)
            vector_store.create_index()  # Create index if it doesn't exist
        else:
            vector_store = ChromaVectorStore()
            vector_store.create_index()
        
        # Initialize retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            top_k=config["top_k"],
            search_type=config["search_type"]
        )
        
        # Initialize RAG chain
        if config["provider"] == "azure":
            rag_chain = RAGChain(
                retriever=retriever,
                llm_provider="azure",
                azure_endpoint=st.session_state.get("azure_endpoint"),
                azure_api_key=st.session_state.get("azure_key"),
                azure_deployment=st.session_state.get("azure_deployment", "gpt-4")
            )
        elif config["provider"] == "openai":
            rag_chain = RAGChain(
                retriever=retriever,
                llm_provider="openai",
                openai_api_key=st.session_state.get("openai_key")
            )
        else:
            rag_chain = RAGChain(
                retriever=retriever,
                llm_provider="ollama",
                ollama_base_url=st.session_state.get("ollama_url"),
                ollama_model=st.session_state.get("ollama_model")
            )
        
        return rag_chain, vector_store, embedding_gen
        
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        return None, None, None


def handle_document_upload(vector_store, embedding_gen):
    """Handle document upload and ingestion."""
    st.sidebar.subheader("📄 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.sidebar.button("📥 Ingest Documents"):
        with st.spinner("Processing documents..."):
            try:
                from src.document_processor import DocumentLoader, DocumentChunker
                
                loader = DocumentLoader()
                chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
                
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    # Save temporarily
                    temp_path = Path(f"./temp_{uploaded_file.name}")
                    temp_path.write_bytes(uploaded_file.getvalue())
                    
                    # Load and chunk
                    docs = loader.load(str(temp_path))
                    chunks = chunker.chunk(docs)
                    all_chunks.extend(chunks)
                    
                    # Clean up
                    temp_path.unlink()
                
                # Generate embeddings
                texts = [c.content for c in all_chunks]
                embeddings = embedding_gen.embed_batch(texts)
                embedding_vectors = [e.embedding for e in embeddings]
                
                # Add to vector store
                vector_store.add_documents(all_chunks, embedding_vectors)
                
                st.session_state.documents_loaded = True
                st.sidebar.success(f"Ingested {len(all_chunks)} chunks from {len(uploaded_files)} files")
                
            except Exception as e:
                st.sidebar.error(f"Error ingesting documents: {e}")


def display_chat():
    """Display chat interface."""
    st.title("Azure RAG Knowledge Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** (score: {source['score']:.2f})")
                        st.caption(source.get("content_preview", "")[:100] + "...")


def handle_chat_input(rag_chain):
    """Handle chat input and generate response."""
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if rag_chain is None:
                st.error("RAG chain not initialized. Please check your configuration.")
                return
            
            if not st.session_state.documents_loaded:
                st.warning("No documents loaded. Please upload documents first.")
                return
            
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.query(prompt)
                    
                    st.markdown(response.answer)
                    
                    # Show sources
                    if response.sources:
                        with st.expander("Sources"):
                            for source in response.sources:
                                st.markdown(f"- **{source['source']}** (score: {source['score']:.2f})")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")


def display_metrics():
    """Display evaluation metrics (if available)."""
    with st.expander("Status"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Loaded", 
                     "Yes" if st.session_state.documents_loaded else "No")
        
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        with col3:
            st.metric("RAG Status", 
                     "Ready" if st.session_state.rag_chain else "Not Initialized")


def main():
    """Main application."""
    st.set_page_config(
        page_title="Azure RAG Knowledge Assistant",
        layout="wide"
    )
    
    # Initialize
    init_session_state()
    
    # Setup sidebar and get config
    config = setup_sidebar()
    
    # Initialize button
    if st.sidebar.button("Initialize RAG"):
        rag_chain, vector_store, embedding_gen = init_rag_chain(config)
        if rag_chain:
            st.session_state.rag_chain = rag_chain
            st.session_state.vector_store = vector_store
            st.session_state.embedding_gen = embedding_gen
            st.sidebar.success("RAG initialized")
    
    # Handle document upload if initialized
    if st.session_state.get("vector_store") and st.session_state.get("embedding_gen"):
        handle_document_upload(
            st.session_state.vector_store,
            st.session_state.embedding_gen
        )
    
    # Main chat interface
    display_chat()
    display_metrics()
    
    # Handle input
    handle_chat_input(st.session_state.rag_chain)


if __name__ == "__main__":
    main()
