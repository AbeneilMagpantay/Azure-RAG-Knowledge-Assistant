"""Azure RAG Knowledge Assistant - Streamlit UI."""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "rag_chain": None,
        "documents_loaded": False,
        "vector_store": None,
        "embedding_gen": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_sidebar():
    """Configure sidebar settings and return config dict."""
    st.sidebar.title("Configuration")
    
    provider = st.sidebar.selectbox("LLM Provider", ["ollama", "azure", "openai"])
    
    if provider == "azure":
        st.sidebar.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""), key="azure_endpoint", type="password")
        st.sidebar.text_input("Azure API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), key="azure_key", type="password")
        st.sidebar.text_input("Deployment", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"), key="azure_deployment")
    elif provider == "openai":
        st.sidebar.text_input("OpenAI Key", value=os.getenv("OPENAI_API_KEY", ""), key="openai_key", type="password")
    else:
        st.sidebar.text_input("Ollama URL", value="http://localhost:11434", key="ollama_url")
        st.sidebar.text_input("Ollama Model", value="llama3", key="ollama_model")
    
    st.sidebar.subheader("Search")
    top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
    search_type = st.sidebar.selectbox("Search Type", ["hybrid", "vector", "keyword"])
    
    st.sidebar.subheader("Storage")
    use_azure_search = st.sidebar.checkbox("Use Azure AI Search", value=bool(os.getenv("AZURE_SEARCH_ENDPOINT")))
    
    return {"provider": provider, "top_k": top_k, "search_type": search_type, "use_azure_search": use_azure_search}


def init_rag_chain(config):
    """Initialize RAG chain with current configuration."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.vectorstore.azure_search import AzureSearchVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        # Use Azure embeddings when Azure Search is enabled
        if config["use_azure_search"]:
            embedding_gen = EmbeddingGenerator(provider="azure")
            vector_store = AzureSearchVectorStore(embedding_dimension=3072)
            vector_store.create_index()
        else:
            embedding_gen = EmbeddingGenerator(provider="local")
            vector_store = ChromaVectorStore()
            vector_store.create_index()
        
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_gen,
            top_k=config["top_k"],
            search_type=config["search_type"]
        )
        
        # Initialize RAG chain based on provider
        llm_config = {"retriever": retriever, "llm_provider": config["provider"]}
        
        if config["provider"] == "azure":
            llm_config.update({
                "azure_endpoint": st.session_state.get("azure_endpoint"),
                "azure_api_key": st.session_state.get("azure_key"),
                "azure_deployment": st.session_state.get("azure_deployment", "gpt-4")
            })
        elif config["provider"] == "openai":
            llm_config["openai_api_key"] = st.session_state.get("openai_key")
        else:
            llm_config.update({
                "ollama_base_url": st.session_state.get("ollama_url"),
                "ollama_model": st.session_state.get("ollama_model")
            })
        
        return RAGChain(**llm_config), vector_store, embedding_gen
        
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None, None, None


def handle_document_upload(vector_store, embedding_gen):
    """Handle document upload and ingestion."""
    st.sidebar.subheader("Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.sidebar.button("Ingest Documents"):
        with st.spinner("Processing..."):
            try:
                from src.document_processor import DocumentLoader, DocumentChunker
                
                loader = DocumentLoader()
                chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    temp_path = Path(f"./temp_{uploaded_file.name}")
                    temp_path.write_bytes(uploaded_file.getvalue())
                    
                    docs = loader.load(str(temp_path))
                    chunks = chunker.chunk(docs)
                    all_chunks.extend(chunks)
                    temp_path.unlink()
                
                texts = [c.content for c in all_chunks]
                embeddings = embedding_gen.embed_batch(texts)
                embedding_vectors = [e.embedding for e in embeddings]
                
                vector_store.add_documents(all_chunks, embedding_vectors)
                st.session_state.documents_loaded = True
                st.sidebar.success(f"Ingested {len(all_chunks)} chunks")
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")


def display_chat():
    """Display chat interface."""
    st.title("Azure RAG Knowledge Assistant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")


def handle_chat_input(rag_chain):
    """Handle chat input and generate response."""
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not rag_chain:
                st.error("RAG not initialized. Click 'Initialize RAG' first.")
                return
            
            if not st.session_state.documents_loaded:
                st.warning("No documents loaded. Upload documents first.")
                return
            
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.query(prompt)
                    st.markdown(response.answer)
                    
                    if response.sources:
                        with st.expander("Sources"):
                            for src in response.sources:
                                st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")


def display_status():
    """Display system status."""
    with st.expander("Status"):
        cols = st.columns(3)
        cols[0].metric("Documents Loaded", "Yes" if st.session_state.documents_loaded else "No")
        cols[1].metric("Messages", len(st.session_state.messages))
        cols[2].metric("RAG Status", "Ready" if st.session_state.rag_chain else "Not Initialized")


def main():
    """Main application entry point."""
    st.set_page_config(page_title="Azure RAG Knowledge Assistant", layout="wide")
    
    init_session_state()
    config = setup_sidebar()
    
    if st.sidebar.button("Initialize RAG"):
        rag_chain, vector_store, embedding_gen = init_rag_chain(config)
        if rag_chain:
            st.session_state.rag_chain = rag_chain
            st.session_state.vector_store = vector_store
            st.session_state.embedding_gen = embedding_gen
            st.sidebar.success("RAG initialized")
    
    if st.session_state.vector_store and st.session_state.embedding_gen:
        handle_document_upload(st.session_state.vector_store, st.session_state.embedding_gen)
    
    display_chat()
    display_status()
    handle_chat_input(st.session_state.rag_chain)


if __name__ == "__main__":
    main()
