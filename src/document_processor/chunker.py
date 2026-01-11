"""Document chunking with various strategies."""

from typing import List, Optional
from dataclasses import dataclass, field

from .loader import Document


@dataclass
class Chunk:
    """A chunk of text from a document with metadata."""
    
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            # Generate a simple chunk ID
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            source = self.metadata.get("source", "unknown")
            chunk_index = self.metadata.get("chunk_index", 0)
            self.chunk_id = f"{source}_{chunk_index}_{content_hash}"


class DocumentChunker:
    """
    Split documents into chunks for embedding and retrieval.
    
    Strategies:
    - Fixed size: Split by character/token count with overlap
    - Semantic: Split by paragraphs/sections (preserves context)
    - Recursive: Hierarchical splitting for optimal chunk sizes
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        length_function: str = "characters"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy ("fixed", "semantic", "recursive")
            length_function: How to measure length ("characters" or "tokens")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.length_function = length_function
        
        # Initialize tokenizer for token-based chunking
        self._tokenizer = None
        if length_function == "tokens":
            self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tiktoken for accurate token counting."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except ImportError:
            print("Warning: tiktoken not installed. Using character-based chunking.")
            self.length_function = "characters"
    
    def _get_length(self, text: str) -> int:
        """Get the length of text based on configured length function."""
        if self.length_function == "tokens" and self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text)
    
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            if self.strategy == "fixed":
                chunks = self._fixed_chunk(doc)
            elif self.strategy == "semantic":
                chunks = self._semantic_chunk(doc)
            else:  # recursive (default)
                chunks = self._recursive_chunk(doc)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _fixed_chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        text = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_strategy": "fixed"
                    }
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _semantic_chunk(self, document: Document) -> List[Chunk]:
        """Split document by semantic boundaries (paragraphs, sections)."""
        text = document.content
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # Check if adding this paragraph exceeds chunk size
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if self._get_length(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunk_strategy": "semantic"
                        }
                    ))
                    chunk_index += 1
                
                # Handle paragraphs larger than chunk_size
                if self._get_length(para) > self.chunk_size:
                    # Fall back to fixed chunking for this paragraph
                    sub_chunks = self._split_large_text(para, document.metadata, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_strategy": "semantic"
                }
            ))
        
        return chunks
    
    def _recursive_chunk(self, document: Document) -> List[Chunk]:
        """
        Recursively split document using multiple separators.
        Tries to split on natural boundaries first.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(
            document.content,
            separators,
            document.metadata,
            chunk_index=0
        )
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        metadata: dict,
        chunk_index: int
    ) -> List[Chunk]:
        """Recursively split text using decreasing separator granularity."""
        chunks = []
        
        if not separators:
            # Base case: no more separators, just split by size
            return self._split_large_text(text, metadata, chunk_index)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            # Empty separator means split by characters
            parts = list(text)
        else:
            parts = text.split(separator)
        
        current_chunk = ""
        
        for i, part in enumerate(parts):
            # Reconstruct with separator for accurate text
            if i > 0 and separator:
                test_chunk = current_chunk + separator + part
            else:
                test_chunk = current_chunk + part if current_chunk else part
            
            if self._get_length(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk,
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index + len(chunks),
                            "chunk_strategy": "recursive"
                        }
                    ))
                
                # Process the part that didn't fit
                if self._get_length(part) > self.chunk_size:
                    # Part is too large, recurse with smaller separators
                    sub_chunks = self._recursive_split(
                        part,
                        remaining_separators,
                        metadata,
                        chunk_index + len(chunks)
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk,
                metadata={
                    **metadata,
                    "chunk_index": chunk_index + len(chunks),
                    "chunk_strategy": "recursive"
                }
            ))
        
        return chunks
    
    def _split_large_text(
        self,
        text: str,
        metadata: dict,
        start_index: int
    ) -> List[Chunk]:
        """Split text that exceeds chunk_size into fixed chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_index": start_index + len(chunks),
                        "chunk_strategy": "recursive_fallback"
                    }
                ))
            
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks
