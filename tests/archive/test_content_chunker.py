"""Tests for Content Chunking with Context Inheritance"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from core.content_chunker import ContentChunker, StructureAwareChunker
from core.chunking import (
    ContextualChunk, 
    ContextGroup, 
    ChunkingResult,
    ChunkingStrategy,
    ContextGroupType
)
from core.parsers import Document, DocumentMetadata, DocumentType, Segment


@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    metadata = DocumentMetadata(
        title="Test Document",
        document_type=DocumentType.TXT,
        page_count=1,
        created_at=datetime.now()
    )
    
    segments = [
        TextSegment(
            content="Introduction to Machine Learning. This is the first section.",
            metadata={"type": "header", "level": 1},
            page_or_slide=1,
            position=(0, 50)
        ),
        TextSegment(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"type": "paragraph"},
            page_or_slide=1,
            position=(51, 150)
        ),
        TextSegment(
            content="Deep Learning Methods. Neural networks are the foundation.",
            metadata={"type": "header", "level": 2},
            page_or_slide=1,
            position=(151, 200)
        ),
        TextSegment(
            content="Deep learning uses multiple layers to progressively extract features from raw input.",
            metadata={"type": "paragraph"},
            page_or_slide=1,
            position=(201, 300)
        )
    ]
    
    return Document(
        metadata=metadata,
        segments=segments,
        visual_elements=[]
    )


@pytest.fixture
def chunker_config():
    """Configuration for testing chunker"""
    return {
        "chunking": {
            "strategies": {
                "txt": {
                    "max_tokens": 100,
                    "min_tokens": 20,
                    "overlap_tokens": 10,
                    "respect_boundaries": True
                }
            },
            "enable_context_inheritance": True,
            "performance": {
                "enable_async_processing": False,  # Sync for easier testing
                "max_concurrent_groups": 1
            }
        },
        "context_inheritance": {
            "enabled": True,
            "max_context_tokens": 50,
            "llm": {
                "model": "test-model",
                "temperature": 0.1
            }
        }
    }


class TestContentChunker:
    """Test Content Chunker functionality"""
    
    def test_chunker_initialization(self, chunker_config):
        """Test that chunker initializes correctly"""
        chunker = ContentChunker(chunker_config)
        
        assert chunker.config == chunker_config
        assert chunker.enable_context_inheritance is True
        assert chunker.base_chunker is not None
        assert chunker.context_grouper is not None
        assert chunker.context_summarizer is not None

    @pytest.mark.asyncio
    async def test_basic_chunking_without_context_inheritance(self, sample_document, chunker_config):
        """Test basic chunking without LLM context inheritance"""
        # Disable context inheritance for this test
        chunker_config["chunking"]["enable_context_inheritance"] = False
        chunker = ContentChunker(chunker_config)
        
        result = await chunker.chunk_document(sample_document)
        
        assert isinstance(result, ChunkingResult)
        assert result.document_id == "Test Document"
        assert result.source_document == sample_document
        assert len(result.contextual_chunks) > 0
        assert len(result.context_groups) > 0
        assert result.chunking_strategy == ChunkingStrategy.STRUCTURE_AWARE

    @pytest.mark.asyncio
    async def test_chunk_content_integration(self, sample_document, chunker_config):
        """Test that chunks contain correct content"""
        chunker = ContentChunker(chunker_config)
        
        result = await chunker.chunk_document(sample_document)
        
        # Check that content from segments is preserved in chunks
        all_chunk_content = " ".join(chunk.content for chunk in result.contextual_chunks)
        assert "Machine Learning" in all_chunk_content
        assert "artificial intelligence" in all_chunk_content
        assert "Deep Learning" in all_chunk_content
        assert "Neural networks" in all_chunk_content

    @pytest.mark.asyncio
    async def test_context_group_formation(self, sample_document, chunker_config):
        """Test that context groups are formed correctly"""
        chunker = ContentChunker(chunker_config)
        
        result = await chunker.chunk_document(sample_document)
        
        assert len(result.context_groups) > 0
        
        # Check group properties
        for group in result.context_groups:
            assert isinstance(group, ContextGroup)
            assert group.group_id is not None
            assert group.document_id == "Test Document"
            assert len(group.chunks) > 0
            assert group.group_type in [ContextGroupType.TOPIC, ContextGroupType.SECTION, ContextGroupType.PAGE_RANGE]

    @pytest.mark.asyncio
    async def test_chunk_statistics(self, sample_document, chunker_config):
        """Test that statistics are calculated correctly"""
        chunker = ContentChunker(chunker_config)
        
        result = await chunker.chunk_document(sample_document)
        
        assert result.processing_stats is not None
        assert result.processing_stats.total_chunks > 0
        assert result.processing_stats.processing_time_seconds > 0
        
        # Test total properties
        assert result.total_chunks == len(result.contextual_chunks)
        assert result.total_groups == len(result.context_groups)

    @pytest.mark.asyncio  
    async def test_multiple_documents_processing(self, sample_document, chunker_config):
        """Test processing multiple documents"""
        chunker = ContentChunker(chunker_config)
        
        # Create second document
        metadata2 = DocumentMetadata(
            title="Second Document",
            document_type=DocumentType.TXT,
            page_count=1,
            created_at=datetime.now()
        )
        
        document2 = Document(
            metadata=metadata2,
            segments=[
                TextSegment(
                    content="This is the second document about data science.",
                    metadata={"type": "paragraph"},
                    page_or_slide=1,
                    position=(0, 50)
                )
            ],
            visual_elements=[]
        )
        
        results = await chunker.chunk_multiple_documents([sample_document, document2])
        
        assert len(results) == 2
        assert all(isinstance(result, ChunkingResult) for result in results)
        assert results[0].document_id == "Test Document"
        assert results[1].document_id == "Second Document"

    def test_chunking_statistics_aggregation(self, sample_document, chunker_config):
        """Test aggregation of chunking statistics"""
        chunker = ContentChunker(chunker_config)
        
        # Create mock results
        result1 = ChunkingResult(
            document_id="doc1",
            source_document=sample_document,
            contextual_chunks=[
                ContextualChunk(chunk_id="c1", content="chunk1", generates_context=True),
                ContextualChunk(chunk_id="c2", content="chunk2", generates_context=False)
            ],
            context_groups=[],
            chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            processing_stats=None,
            processing_config={}
        )
        
        result2 = ChunkingResult(
            document_id="doc2",
            source_document=sample_document,
            contextual_chunks=[
                ContextualChunk(chunk_id="c3", content="chunk3", generates_context=True)
            ],
            context_groups=[],
            chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            processing_stats=None,
            processing_config={}
        )
        
        stats = chunker.get_chunking_stats([result1, result2])
        
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 3
        assert stats["chunks_with_context"] == 2  # Two chunks generate context
        assert stats["avg_chunks_per_doc"] == 1.5

    @pytest.mark.asyncio
    async def test_error_handling_empty_document(self, chunker_config):
        """Test handling of empty document"""
        chunker = ContentChunker(chunker_config)
        
        # Create empty document
        empty_metadata = DocumentMetadata(
            title="Empty Document",
            document_type=DocumentType.TXT,
            page_count=0,
            created_at=datetime.now()
        )
        
        empty_document = Document(
            metadata=empty_metadata,
            segments=[],
            visual_elements=[]
        )
        
        result = await chunker.chunk_document(empty_document)
        
        assert isinstance(result, ChunkingResult)
        assert len(result.contextual_chunks) == 0
        assert len(result.context_groups) == 0


class TestStructureAwareChunker:
    """Test Structure-Aware Chunking logic"""
    
    def test_structure_aware_chunker_initialization(self, chunker_config):
        """Test chunker initialization"""
        chunker = StructureAwareChunker(chunker_config)
        assert chunker.config == chunker_config

    def test_chunk_document_creation(self, sample_document, chunker_config):
        """Test basic chunk creation"""
        chunker = StructureAwareChunker(chunker_config)
        strategy_config = chunker_config["chunking"]["strategies"]["txt"]
        
        chunks = chunker.chunk_document(sample_document, strategy_config)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ContextualChunk) for chunk in chunks)
        
        # Check that chunks have required properties
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.content is not None
            assert len(chunk.content) > 0

    def test_structure_preservation(self, sample_document, chunker_config):
        """Test that document structure is preserved"""
        chunker = StructureAwareChunker(chunker_config)
        strategy_config = chunker_config["chunking"]["strategies"]["txt"]
        
        chunks = chunker.chunk_document(sample_document, strategy_config)
        
        # Check that chunks maintain positional information
        for chunk in chunks:
            assert hasattr(chunk, 'page_range') or hasattr(chunk, 'position_range')
            assert chunk.document_id is not None

    def test_visual_elements_integration(self, chunker_config):
        """Test integration of visual elements"""
        chunker = StructureAwareChunker(chunker_config)
        strategy_config = chunker_config["chunking"]["strategies"]["txt"]
        
        # Create document with visual elements
        from core.parsers import VisualElement, VisualElementType
        
        visual = VisualElement(
            element_type=VisualElementType.IMAGE,
            page_or_slide=1,
            position=(100, 200),
            size=(50, 50),
            description="Test image",
            vlm_description="A test image showing data visualization"
        )
        
        metadata = DocumentMetadata(
            title="Document with Visuals",
            document_type=DocumentType.TXT,
            page_count=1,
            created_at=datetime.now()
        )
        
        document = Document(
            metadata=metadata,
            segments=[
                TextSegment(
                    content="This document contains visual elements.",
                    metadata={"type": "paragraph"},
                    page_or_slide=1,
                    position=(0, 100)
                )
            ],
            visual_elements=[visual]
        )
        
        chunks = chunker.chunk_document(document, strategy_config)
        
        assert len(chunks) > 0
        # Check that visual elements are associated with chunks
        chunk_with_visual = next((chunk for chunk in chunks if chunk.visual_elements), None)
        if chunk_with_visual:
            assert len(chunk_with_visual.visual_elements) > 0
            assert chunk_with_visual.visual_elements[0].vlm_description == "A test image showing data visualization"


@pytest.mark.asyncio
async def test_content_chunker_with_mock_llm(sample_document, chunker_config):
    """Test content chunker with mocked LLM for context inheritance"""
    
    # Mock the context summarizer's LLM calls
    with patch('core.chunking.context_summarizer.ContextSummarizer.generate_context_summary') as mock_generate:
        mock_generate.return_value = "Mocked context summary about machine learning concepts"
        
        with patch('core.chunking.context_summarizer.ContextSummarizer.process_context_inheritance') as mock_process:
            # Create mock processed chunks
            processed_chunks = [
                ContextualChunk(
                    chunk_id="test_chunk_1",
                    content="Machine learning is a subset of AI",
                    inherited_context="Mocked context about AI",
                    generates_context=True
                ),
                ContextualChunk(
                    chunk_id="test_chunk_2", 
                    content="Deep learning uses neural networks",
                    inherited_context="Mocked context about AI",
                    generates_context=False
                )
            ]
            mock_process.return_value = processed_chunks
            
            chunker = ContentChunker(chunker_config)
            
            result = await chunker.chunk_document_with_context_inheritance(
                sample_document, 
                "Extract key concepts and relationships from this text about machine learning."
            )
            
            assert isinstance(result, ChunkingResult)
            assert len(result.contextual_chunks) > 0
            
            # Verify that context inheritance was attempted
            if chunker_config["chunking"]["enable_context_inheritance"]:
                mock_process.assert_called()
                
                # Check that chunks have inherited context
                chunks_with_context = [c for c in result.contextual_chunks if c.inherited_context]
                assert len(chunks_with_context) > 0