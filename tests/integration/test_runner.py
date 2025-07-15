#!/usr/bin/env python3
"""Test runner script for the Generic Knowledge Graph Pipeline System

This script provides different test execution modes without requiring external dependencies.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import Config, get_config
from core.content_chunker import ContentChunker
from core.batch_processor import BatchProcessor
from plugins.parsers.parser_factory import ParserFactory
from plugins.parsers.base_parser import Document, DocumentMetadata, DocumentType, Segment


class TestResult:
    """Simple test result container"""
    def __init__(self, name: str, passed: bool, error: str = None, duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.error = error
        self.duration = duration


class TestRunner:
    """Simple test runner for basic functionality testing"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        
    def run_test(self, test_func, name: str) -> TestResult:
        """Run a single test function"""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            
            duration = time.time() - start_time
            result = TestResult(name, True, duration=duration)
            print(f"✅ {name} ({duration:.3f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            if isinstance(e, AssertionError):
                error_msg = str(e) if str(e) else "Assertion failed (no message)"
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
            result = TestResult(name, False, error_msg, duration=duration)
            print(f"❌ {name} ({duration:.3f}s): {error_msg}")
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.duration for r in self.results)
        
        print(f"\n📊 Test Summary:")
        print(f"   Passed: {passed}/{total}")
        print(f"   Failed: {total - passed}/{total}")
        print(f"   Total time: {total_time:.3f}s")
        
        if total - passed > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   - {result.name}: {result.error}")


def create_sample_document() -> Document:
    """Create a sample document for testing"""
    metadata = DocumentMetadata(
        title="Test Document",
        document_type=DocumentType.TXT,
        page_count=1,
        created_date=datetime.now()
    )
    
    segments = [
        Segment(
            content="Introduction to Machine Learning. This is an introductory section about AI.",
            page_number=1,
            segment_index=0,
            segment_type="header",
            metadata={"type": "header", "level": 1}
        ),
        Segment(
            content="Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions or decisions based on data.",
            page_number=1,
            segment_index=1,
            segment_type="paragraph",
            metadata={"type": "paragraph"}
        ),
        Segment(
            content="Deep Learning Methods. Neural networks form the foundation of deep learning.",
            page_number=1,
            segment_index=2,
            segment_type="header",
            metadata={"type": "header", "level": 2}
        ),
        Segment(
            content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input data.",
            page_number=1,
            segment_index=3,
            segment_type="paragraph",
            metadata={"type": "paragraph"}
        )
    ]
    
    return Document(
        content=" ".join(seg.content for seg in segments),
        metadata=metadata,
        segments=segments,
        visual_elements=[]
    )


def test_config_loading():
    """Test configuration loading"""
    config = get_config()
    assert config is not None
    assert hasattr(config, 'domain')
    assert hasattr(config, 'chunking')
    assert hasattr(config, 'llm')


def test_parser_factory():
    """Test parser factory functionality"""
    factory = ParserFactory()
    
    # Test parser creation for different file types
    from pathlib import Path
    
    pdf_path = Path("test.pdf")
    pdf_parser = factory.get_parser_for_file(pdf_path)
    assert pdf_parser is not None
    
    docx_path = Path("test.docx")
    docx_parser = factory.get_parser_for_file(docx_path)
    assert docx_parser is not None
    
    xlsx_path = Path("test.xlsx")
    xlsx_parser = factory.get_parser_for_file(xlsx_path)
    assert xlsx_parser is not None
    
    pptx_path = Path("test.pptx")
    pptx_parser = factory.get_parser_for_file(pptx_path)
    assert pptx_parser is not None


async def test_content_chunker_basic():
    """Test basic content chunker functionality"""
    # Create test config
    config = {
        "chunking": {
            "strategies": {
                "txt": {
                    "max_tokens": 150,
                    "min_tokens": 50,
                    "overlap_tokens": 20,
                    "respect_boundaries": True
                }
            },
            "enable_context_inheritance": False,  # Disable LLM calls for this test
            "performance": {
                "enable_async_processing": False,
                "max_concurrent_groups": 1
            }
        }
    }
    
    chunker = ContentChunker(config)
    document = create_sample_document()
    
    result = await chunker.chunk_document(document)
    
    assert result is not None
    assert result.document_id == "Test Document"
    assert len(result.contextual_chunks) > 0
    assert len(result.context_groups) > 0
    assert result.processing_stats is not None


async def test_batch_processor():
    """Test batch processor functionality"""
    config = get_config()
    
    # Create simple config dict for batch processor
    batch_config = {
        "max_concurrent": 2,
        "timeout": 30,
        "retry_attempts": 3
    }
    
    processor = BatchProcessor(batch_config)
    
    # Test initialization
    assert processor is not None
    assert not processor.is_processing


def test_document_structure():
    """Test document data structure"""
    document = create_sample_document()
    
    assert document.metadata.title == "Test Document"
    assert document.metadata.document_type == DocumentType.TXT
    assert len(document.segments) == 4
    
    # Check segments
    intro_segment = document.segments[0]
    assert "Introduction to Machine Learning" in intro_segment.content
    assert intro_segment.metadata["type"] == "header"
    
    para_segment = document.segments[1] 
    assert "subset of artificial intelligence" in para_segment.content
    assert para_segment.metadata["type"] == "paragraph"


def test_chunking_models():
    """Test chunking data models"""
    from core.chunking import ContextualChunk, ContextGroup, ContextGroupType, ChunkType
    
    # Test chunk creation
    chunk = ContextualChunk(
        chunk_id="test_chunk_1",
        source_document_id="test_doc",
        content="This is test chunk content",
        token_count=5,
        chunk_type=ChunkType.CONTENT
    )
    
    assert chunk.chunk_id == "test_chunk_1"
    assert chunk.content == "This is test chunk content"
    assert chunk.source_document_id == "test_doc"
    
    # Test context group
    group = ContextGroup(
        group_id="test_group_1",
        document_id="test_doc",
        group_type=ContextGroupType.TOPIC,
        chunks=[chunk]
    )
    
    assert group.group_id == "test_group_1"
    assert len(group.chunks) == 1
    assert group.group_type == ContextGroupType.TOPIC


def test_llm_client_initialization():
    """Test LLM client initialization"""
    from core.clients.hochschul_llm import HochschulLLMClient
    from core.clients.vllm_smoldocling import VLLMSmolDoclingClient
    from core.clients.qwen25_vl import Qwen25VLClient
    
    # Test client instantiation (without actual API calls)
    try:
        hochschul_client = HochschulLLMClient()
        assert hochschul_client is not None
    except Exception as e:
        # Expected if credentials not configured
        assert "credentials" in str(e).lower() or "endpoint" in str(e).lower()
    
    try:
        vllm_client = VLLMSmolDoclingClient()
        assert vllm_client is not None
    except Exception as e:
        # Expected if endpoint not configured
        assert "endpoint" in str(e).lower() or "url" in str(e).lower()
    
    try:
        qwen_client = Qwen25VLClient()
        assert qwen_client is not None
    except Exception as e:
        # Expected if credentials not configured
        assert "credentials" in str(e).lower() or "endpoint" in str(e).lower()


async def test_content_processing_pipeline():
    """Test the full content processing pipeline (without LLM calls)"""
    # Mock configuration without LLM dependencies
    config = {
        "chunking": {
            "strategies": {
                "txt": {
                    "max_tokens": 50,  # Smaller chunks to force multiple chunks
                    "min_tokens": 20,
                    "overlap_tokens": 10,
                    "respect_boundaries": True
                }
            },
            "enable_context_inheritance": False,  # No LLM calls
            "performance": {
                "enable_async_processing": False,
                "max_concurrent_groups": 1
            }
        }
    }
    
    # Test the pipeline
    chunker = ContentChunker(config)
    document = create_sample_document()
    
    # Process document
    result = await chunker.chunk_document(document)
    
    # Verify results
    assert result.document_id == "Test Document"
    assert len(result.contextual_chunks) >= 1
    
    # Check chunk content preservation
    all_content = " ".join(chunk.content for chunk in result.contextual_chunks)
    assert "Machine learning" in all_content
    assert "Deep learning" in all_content
    assert "artificial intelligence" in all_content
    
    # Check that chunks have proper structure
    if result.contextual_chunks:
        first_chunk = result.contextual_chunks[0]
        assert hasattr(first_chunk, 'chunk_id'), "Chunk missing chunk_id attribute"
        assert hasattr(first_chunk, 'content'), "Chunk missing content attribute"
        assert hasattr(first_chunk, 'source_document_id'), "Chunk missing source_document_id attribute"
        assert len(first_chunk.content) > 0, "Chunk has empty content"


def test_error_handling():
    """Test error handling in various components"""
    # Test parser factory with invalid type
    factory = ParserFactory()
    
    try:
        from pathlib import Path
        factory.get_parser_for_file(Path("invalid.unknown"))
        assert False, "Should have raised an error"
    except Exception as e:
        assert len(str(e)) > 0  # Should have error message
    
    # Test chunker with invalid config
    try:
        ContentChunker({})  # Empty config
        # Should either work with defaults or raise meaningful error
    except Exception as e:
        assert len(str(e)) > 0  # Should have error message


def main():
    """Main test execution"""
    print("🧪 Generic Knowledge Graph Pipeline - Test Runner")
    print("=" * 60)
    print("Testing implemented components without external dependencies...")
    print()
    
    runner = TestRunner()
    
    # Run basic tests
    runner.run_test(test_config_loading, "Configuration Loading")
    runner.run_test(test_parser_factory, "Parser Factory")
    runner.run_test(test_document_structure, "Document Data Structure")
    runner.run_test(test_chunking_models, "Chunking Models")
    runner.run_test(test_llm_client_initialization, "LLM Client Initialization")
    runner.run_test(test_content_chunker_basic, "Content Chunker Basic")
    runner.run_test(test_batch_processor, "Batch Processor")
    runner.run_test(test_content_processing_pipeline, "Content Processing Pipeline")
    runner.run_test(test_error_handling, "Error Handling")
    
    # Print summary
    runner.print_summary()
    
    # Return exit code
    failed_tests = sum(1 for r in runner.results if not r.passed)
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)