# Azure RAG Knowledge Assistant - Sample Document

This is a sample document to test the RAG pipeline.

## What is Azure OpenAI?

Azure OpenAI is a cloud service that provides access to OpenAI's powerful language models through Microsoft Azure. It offers enterprise-grade security, compliance, and integration capabilities.

Key features include:
- Access to GPT-4, GPT-3.5, and embedding models
- Enterprise security and compliance (SOC2, GDPR, HIPAA)
- Private networking and data isolation
- Content filtering and responsible AI guardrails

## How to Use Azure OpenAI

To use Azure OpenAI, you need to:

1. Create an Azure OpenAI resource in the Azure portal
2. Deploy a model (like GPT-4 or text-embedding-3-large)
3. Get your endpoint and API key
4. Call the API using the OpenAI SDK with Azure configuration

Example code:
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key",
    api_version="2024-02-15-preview"
)

response = client.chat.completions.create(
    model="your-deployment-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Azure AI Search

Azure AI Search (formerly Cognitive Search) is an enterprise search service with AI capabilities.

Features:
- Vector search for semantic similarity
- Keyword search for exact matching
- Hybrid search combining both approaches
- Built-in AI enrichment pipelines
- Semantic ranking for improved relevance

## RAG Architecture

Retrieval-Augmented Generation (RAG) is an architecture pattern that:

1. Retrieves relevant documents from a knowledge base
2. Augments the LLM prompt with retrieved context
3. Generates responses grounded in the retrieved documents

Benefits:
- Answers based on your specific documents
- Reduced hallucination through grounding
- Up-to-date information (not limited to training data)
- Source citations for trustworthiness
