
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Cookie controller for browser session persistence
try:
    from streamlit_cookies_controller import CookieController
    cookie_controller = CookieController()
except ImportError:
    cookie_controller = None

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
def initialize_rag(provider_config):
    """Initialize RAG components with caching."""
    try:
        from src.embeddings import EmbeddingGenerator
        from src.vectorstore.local_store import ChromaVectorStore
        from src.rag import RAGRetriever, RAGChain
        
        provider = provider_config.get("provider", "local")
        
        # Initialize embedding generator
        embedding_provider = "local" if provider == "ollama" else provider
        embedding_params = {"provider": embedding_provider}
        if provider == "google":
            embedding_params["google_api_key"] = provider_config.get("google_api_key")
            embedding_params["model"] = "models/text-embedding-004"
        elif provider == "openai":
            embedding_params["openai_api_key"] = provider_config.get("openai_api_key")
        elif provider == "azure":
            embedding_params["azure_endpoint"] = provider_config.get("azure_endpoint")
            embedding_params["azure_api_key"] = provider_config.get("azure_api_key")
            embedding_params["azure_deployment"] = provider_config.get("azure_embedding_deployment")
            embedding_params["azure_api_version"] = provider_config.get("azure_api_version")
            
        embedding_gen = EmbeddingGenerator(**embedding_params)
        
        # Initialize vector store
        vector_store = ChromaVectorStore()
        vector_store.create_index()
        
        # Initialize retriever
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_gen
        )
        
        # Initialize RAG chain
        rag_params = {"retriever": retriever, "llm_provider": provider}
        if provider == "google":
            rag_params["google_api_key"] = provider_config.get("google_api_key")
            rag_params["model"] = "gemini-2.0-flash"
        elif provider == "openai":
            rag_params["openai_api_key"] = provider_config.get("openai_api_key")
        elif provider == "azure":
            rag_params["azure_endpoint"] = provider_config.get("azure_endpoint")
            rag_params["azure_api_key"] = provider_config.get("azure_api_key")
            rag_params["azure_deployment"] = provider_config.get("azure_llm_deployment")
            rag_params["azure_api_version"] = provider_config.get("azure_api_version")
            
        rag_chain = RAGChain(**rag_params)
            
        return rag_chain, vector_store, embedding_gen
        
    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return None, None, None

def main():
    st.title("📚 RAG Knowledge Assistant")
    
    # Use fixed session ID for now (cookies don't persist in Streamlit Cloud)
    # TODO: Implement proper user authentication for separate sessions
    SESSION_ID = "global"
    
    # Initialize chat history store
    if "chat_store" not in st.session_state:
        try:
            from src.storage import ChatHistoryStore
            st.session_state.chat_store = ChatHistoryStore(session_id=SESSION_ID)
        except Exception:
            st.session_state.chat_store = None
    
    # Load messages from Supabase or initialize empty
    if "messages" not in st.session_state:
        if st.session_state.chat_store and st.session_state.chat_store.is_connected:
            st.session_state.messages = st.session_state.chat_store.load_history()
        else:
            st.session_state.messages = []
        
    with st.sidebar:
        st.header("Settings")
        
        # Show Supabase connection status
        if st.session_state.chat_store and st.session_state.chat_store.is_connected:
            st.success("💾 Chat history: Connected")
        else:
            st.warning("💾 Chat history: Not connected")
        
        # Provider Selection
        provider = st.selectbox(
            "Select LLM Provider",
            ["Google Gemini", "OpenAI", "Azure OpenAI", "Local (Ollama)"],
            index=0
        )
        
        provider_config = {"provider": "local"}
        
        if provider == "Google Gemini":
            provider_config["provider"] = "google"
            # Try to get key from secrets first
            default_key = ""
            try:
                default_key = st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            except Exception:
                default_key = os.getenv("GOOGLE_API_KEY", "")
            
            api_key = st.text_input("Google API Key", value=default_key, type="password")
            provider_config["google_api_key"] = api_key or default_key
                 
        elif provider == "OpenAI":
            provider_config["provider"] = "openai"
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                provider_config["openai_api_key"] = api_key
            else:
                provider_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
                
        elif provider == "Azure OpenAI":
            provider_config["provider"] = "azure"
            
            def sanitize_azure_endpoint(url):
                if not url: return url
                if "openai.azure.com" in url:
                    # Extract base: https://name.openai.azure.com/
                    import re
                    match = re.search(r'(https://[^/]+\.openai\.azure\.com/)', url)
                    return match.group(1) if match else url
                return url

            raw_endpoint = st.text_input("Azure Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
            provider_config["azure_endpoint"] = sanitize_azure_endpoint(raw_endpoint)
            
            if raw_endpoint and raw_endpoint != provider_config["azure_endpoint"]:
                st.caption(f"ℹ️ specific endpoint detected, using base: `{provider_config['azure_endpoint']}`")
                
            provider_config["azure_api_key"] = st.text_input("Azure API Key", type="password", value=os.getenv("AZURE_OPENAI_API_KEY", ""))
            provider_config["azure_llm_deployment"] = st.text_input("LLM Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"))
            provider_config["azure_embedding_deployment"] = st.text_input("Embedding Deployment Name", value=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"))
            provider_config["azure_api_version"] = st.text_input("API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
            
        elif provider == "Local (Ollama)":
             provider_config["provider"] = "ollama"
        
        st.divider()
        st.header("Upload Documents")
        
        # Initialize RAG
        # Only initialize if we have necessary config to avoid errors
        should_init = True
        if provider == "Google Gemini" and not provider_config.get("google_api_key"):
            st.warning("⚠️ Enter Google API Key")
            should_init = False
        elif provider == "OpenAI" and not provider_config.get("openai_api_key"):
            st.warning("⚠️ Enter OpenAI API Key")
            should_init = False
        elif provider == "Azure OpenAI" and not (provider_config.get("azure_endpoint") and provider_config.get("azure_api_key")):
            st.warning("⚠️ Enter Azure details")
            should_init = False
            
        rag_chain = None
        if should_init:
            rag_chain, vector_store, embedding_gen = initialize_rag(provider_config)
            if rag_chain:
                st.success(f"Connected to {provider}")
        
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, DOCX", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md']
        )
        
        if uploaded_files and st.button("Process Documents"):
            if not rag_chain:
                 st.error("Please configure the provider first.")
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
        st.info("👈 Configure your settings in the sidebar to get started.")
        st.stop()

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.chat_store:
            st.session_state.chat_store.save_message("user", prompt)
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
                if st.session_state.chat_store:
                    st.session_state.chat_store.save_message("assistant", full_response)
                
            except Exception as e:
                st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
