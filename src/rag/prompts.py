"""Prompt templates for RAG pipeline."""

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents.

INSTRUCTIONS:
1. Answer questions ONLY based on the provided context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Always cite your sources using [Source: filename] format
4. Be concise but comprehensive
5. If multiple sources contain relevant information, synthesize them
6. Never make up information not present in the context

RESPONSE FORMAT:
- Provide a clear, direct answer
- Include relevant details from the context
- Cite sources for key facts
"""

# Question-answering prompt template
QA_PROMPT_TEMPLATE = """Based on the following context documents, answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# Prompt for multi-query generation (improves retrieval)
MULTI_QUERY_PROMPT = """You are an AI assistant helping to generate alternative versions of a question.

Given the following question, generate {num_queries} different versions that capture the same intent but use different wording.
This helps retrieve more relevant documents.

Original question: {question}

Generate {num_queries} alternative questions (one per line):"""


# Prompt for query rewriting
QUERY_REWRITE_PROMPT = """You are an AI assistant that rewrites questions to be more specific and searchable.

Rewrite the following question to be clearer and more specific for document retrieval.
Keep the core intent but make it more explicit.

Original question: {question}

Rewritten question:"""


# Prompt for response refinement
REFINEMENT_PROMPT = """You are an AI assistant refining an answer based on additional context.

Original question: {question}

Current answer: {current_answer}

Additional context:
{additional_context}

Provide an improved answer that incorporates the additional context while maintaining accuracy:"""


# Prompt for summarizing retrieved context
CONTEXT_SUMMARY_PROMPT = """Summarize the following context documents into a coherent summary that captures the key information relevant to answering questions.

CONTEXT DOCUMENTS:
{context}

SUMMARY:"""


def format_context(search_results: list) -> str:
    """
    Format search results into a context string for the prompt.
    
    Args:
        search_results: List of SearchResult objects
        
    Returns:
        Formatted context string with source citations
    """
    context_parts = []
    
    for i, result in enumerate(search_results, 1):
        source = result.metadata.get("source", "Unknown")
        page = result.metadata.get("page", "")
        
        # Create source reference
        if page:
            source_ref = f"[Source {i}: {source}, Page {page}]"
        else:
            source_ref = f"[Source {i}: {source}]"
        
        context_parts.append(f"{source_ref}\n{result.content}")
    
    return "\n\n---\n\n".join(context_parts)


def format_qa_prompt(question: str, context: str) -> str:
    """
    Format the full QA prompt.
    
    Args:
        question: User question
        context: Formatted context string
        
    Returns:
        Complete prompt for the LLM
    """
    return QA_PROMPT_TEMPLATE.format(question=question, context=context)
