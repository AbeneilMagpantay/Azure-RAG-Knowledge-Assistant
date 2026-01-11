
import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for modern aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stChatInput {
        border-radius: 20px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    h1 {
        color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        # Check for provider preference
        if os.getenv("GOOGLE_API_KEY"):
            provider = "google"
        elif os.getenv("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "local" # Fallback or Ollama/local
        
        # Initialize embedding generator
        if provider == "google":
            embedding_gen = EmbeddingGenerator(provider="google")
        elif provider == "azure":
            embedding_gen = EmbeddingGenerator(provider="azure")
        elif provider == "openai":
            embedding_gen = EmbeddingGenerator(provider="openai")
        else:
            embedding_gen = EmbeddingGenerator(provider="local")
        
        # Initialize vector store
        vector_store = ChromaVectorStore()
        vector_store.create_index()
        
        # Initialize retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_gen
        )
        
        # Initialize RAG chain
        if provider == "google":
            rag_chain = RAGChain(retriever=retriever, llm_provider="google")
        elif provider == "azure":
            rag_chain = RAGChain(retriever=retriever, llm_provider="azure")
        elif provider == "openai":
            rag_chain = RAGChain(retriever=retriever, llm_provider="openai")
        else:
            rag_chain = RAGChain(retriever=retriever, llm_provider="ollama")
            
        return rag_chain, vector_store, embedding_gen, provider
        
    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return None, None, None, None

def main():
    st.title("📚 RAG Knowledge Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize RAG
    rag_chain, vector_store, embedding_gen, provider = initialize_rag()
    
    if not rag_chain:
        st.warning("Please configure your API keys in .env to continue.")
        st.stop()
        
    with st.sidebar:
        st.header("Upload Documents")
        st.info(f"Using Provider: **{provider.upper()}**")
        
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, DOCX", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md']
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    from src.document_processor import DocumentLoader, DocumentChunker
                    
                    total_chunks = 0
                    for file in uploaded_files:
                        # Save temporarily
                        temp_path = Path(f"./temp_{file.name}")
                        content = file.getvalue()
                        temp_path.write_bytes(content)
                        
                        try:
                            # Load and chunk
                            loader = DocumentLoader()
                            chunker = DocumentChunker()
                            
                            docs = loader.load(str(temp_path))
                            chunks = chunker.chunk(docs)
                            
                            # Generate embeddings
                            texts = [c.content for c in chunks]
                            embeddings = embedding_gen.embed_batch(texts)
                            embedding_vectors = [e.embedding for e in embeddings]
                            
                            # Add to vector store
                            vector_store.add_documents(chunks, embedding_vectors)
                            total_chunks += len(chunks)
                            
                        finally:
                            if temp_path.exists():
                                temp_path.unlink()
                    
                    st.success(f"Successfully processed {total_chunks} chunks from {len(uploaded_files)} files!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        
        if st.button("Clear Knowledge Base"):
            try:
                vector_store.delete_index()
                vector_store.create_index()
                st.success("Knowledge base cleared!")
            except Exception as e:
                st.error(f"Error clearing knowledge base: {e}")

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # Use query_stream for better UX
                full_response = ""
                for token in rag_chain.query_stream(question=prompt):
                     full_response += token
                     message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
