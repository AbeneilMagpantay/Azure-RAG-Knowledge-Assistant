"""Tests for document processor module."""

import pytest
from pathlib import Path

from src.document_processor.loader import DocumentLoader, Document
from src.document_processor.chunker import DocumentChunker, Chunk


class TestDocumentLoader:
    """Tests for DocumentLoader class."""
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.\nIt has multiple lines.")
        
        loader = DocumentLoader()
        documents = loader.load(str(test_file))
        
        assert len(documents) == 1
        assert "test document" in documents[0].content
        assert documents[0].metadata["source"] == str(test_file)
        assert documents[0].metadata["file_type"] == "txt"
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nThis is markdown content.")
        
        loader = DocumentLoader()
        documents = loader.load(str(test_file))
        
        assert len(documents) == 1
        assert "Heading" in documents[0].content
        assert documents[0].metadata["file_type"] == "md"
    
    def test_load_directory(self, tmp_path):
        """Test loading all files from a directory."""
        # Create multiple test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("Content 3")
        
        loader = DocumentLoader()
        documents = loader.load(str(tmp_path))
        
        assert len(documents) == 3
    
    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        chunker = DocumentChunker(
            chunk_size=50,
            chunk_overlap=10,
            strategy="fixed"
        )
        
        doc = Document(
            content="A" * 100,  # 100 characters
            metadata={"source": "test.txt"}
        )
        
        chunks = chunker.chunk([doc])
        
        assert len(chunks) >= 2
        assert all(len(c.content) <= 50 for c in chunks)
    
    def test_semantic_chunking(self):
        """Test semantic chunking by paragraphs."""
        chunker = DocumentChunker(
            chunk_size=200,
            strategy="semantic"
        )
        
        doc = Document(
            content="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            metadata={"source": "test.txt"}
        )
        
        chunks = chunker.chunk([doc])
        
        assert len(chunks) >= 1
    
    def test_recursive_chunking(self):
        """Test recursive chunking."""
        chunker = DocumentChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy="recursive"
        )
        
        doc = Document(
            content="Section 1\n\nParagraph with some content.\n\nSection 2\n\nAnother paragraph with more content.",
            metadata={"source": "test.txt"}
        )
        
        chunks = chunker.chunk([doc])
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_metadata_preserved(self):
        """Test that source metadata is preserved in chunks."""
        chunker = DocumentChunker(chunk_size=50)
        
        doc = Document(
            content="A" * 100,
            metadata={"source": "test.txt", "page": 1}
        )
        
        chunks = chunker.chunk([doc])
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["page"] == 1
            assert "chunk_index" in chunk.metadata
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique."""
        chunker = DocumentChunker(chunk_size=50)
        
        doc = Document(content="A" * 200, metadata={"source": "test.txt"})
        chunks = chunker.chunk([doc])
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_properties(self):
        """Test Document property accessors."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt", "page": 5}
        )
        
        assert doc.source == "test.txt"
        assert doc.page == 5
    
    def test_document_default_metadata(self):
        """Test Document with default metadata."""
        doc = Document(content="Test content")
        
        assert doc.source == "unknown"
        assert doc.page is None
