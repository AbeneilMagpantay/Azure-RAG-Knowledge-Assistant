# Azure RAG Knowledge Assistant

A document Q&A system that lets you chat with your PDFs and documents using Azure AI services.

## What It Does

Upload any document (PDF, DOCX, TXT) and ask questions about it. The system finds relevant sections and generates accurate answers with source citations.

**Built with:**
- Azure OpenAI for embeddings and text generation
- Azure AI Search for fast document retrieval
- Streamlit for the web interface
- Ollama support for local LLM (optional)

## How It Works

```
1. Upload Document → Split into chunks → Generate embeddings → Store in Azure Search
2. Ask Question → Find similar chunks → Send to LLM → Get answer with sources
```

## Quick Start

### Prerequisites

- Python 3.10+
- Azure account with:
  - Azure OpenAI (text-embedding-3-large deployed)
  - Azure AI Search (Free tier works)
- Ollama installed locally (optional, for local LLM)

### Setup

```bash
# Clone the repo
git clone https://github.com/AbeneilMagpantay/Azure-RAG-Knowledge-Assistant.git
cd Azure-RAG-Knowledge-Assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your credentials
cp .env.example .env
# Edit .env with your Azure keys
```

### Run

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501, upload a document, and start asking questions.

## Configuration

Create a `.env` file with your credentials:

```env
# Azure OpenAI (for embeddings)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-admin-key
AZURE_SEARCH_INDEX_NAME=rag-knowledge-index

# LLM (choose one)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Project Structure

```
├── src/
│   ├── document_processor/   # PDF/DOCX loading and chunking
│   ├── embeddings/           # Azure OpenAI embedding generation
│   ├── vectorstore/          # Azure AI Search integration
│   └── rag/                  # Query pipeline and LLM chain
├── app/
│   └── streamlit_app.py      # Web interface
└── tests/                    # Unit tests
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Azure OpenAI (text-embedding-3-large) |
| Vector Store | Azure AI Search |
| LLM | Ollama (llama3) or Azure OpenAI |
| Framework | LangChain |
| UI | Streamlit |

## Features

- **Hybrid Search**: Combines vector similarity with keyword matching for better results
- **Source Citations**: Every answer includes references to the original document sections
- **Multiple File Types**: Supports PDF, DOCX, and TXT files
- **Local LLM Option**: Use Ollama for free, local text generation

## Future Improvements

- [ ] Add conversation memory for follow-up questions
- [ ] Support more file formats (Excel, PowerPoint)
- [ ] Add user authentication
- [ ] Deploy to Azure App Service

## License

MIT
