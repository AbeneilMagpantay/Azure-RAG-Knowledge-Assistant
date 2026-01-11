
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
def initialize_rag(google_api_key=None):
    """Initialize RAG components with caching."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        # Check for provider preference
        if google_api_key:
            provider = "google"
        elif os.getenv("GOOGLE_API_KEY"):
            provider = "google"
        elif os.getenv("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "local" # Fallback or Ollama/local
        
        # Initialize embedding generator
        if provider == "google":
            embedding_gen = EmbeddingGenerator(provider="google", google_api_key=google_api_key)
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
            rag_chain = RAGChain(retriever=retriever, llm_provider="google", google_api_key=google_api_key)
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
        
    with st.sidebar:
        st.header("Settings")
        user_key = st.text_input("Enter Google API Key", type="password", help="Get your key from makersuite.google.com")
        if not user_key and not os.getenv("GOOGLE_API_KEY"):
             st.warning("⚠️ Please provide a Google API Key to use Gemini.")
        
        st.header("Upload Documents")
        
        # Initialize RAG with user key if provided
        rag_chain, vector_store, embedding_gen, provider = initialize_rag(google_api_key=user_key if user_key else None)
        
        if rag_chain:
             st.info(f"Using Provider: **{provider.upper()}**")
        
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, DOCX", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md']
        )
        
        if uploaded_files and st.button("Process Documents"):
            if not rag_chain:
                 st.error("Please provide an API key first.")
            else:
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
            if vector_store:
                try:
                    vector_store.delete_index()
                    vector_store.create_index()
                    st.success("Knowledge base cleared!")
                except Exception as e:
                    st.error(f"Error clearing knowledge base: {e}")

    if not rag_chain:
        st.info("👈 Enter your Google API Key in the sidebar to get started.")
        st.stop()

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
