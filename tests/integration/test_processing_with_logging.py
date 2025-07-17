#!/usr/bin/env python3
"""
Test script for document processing with detailed logging
Shows each step of the processing pipeline
"""

import logging
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import get_config
from core.content_chunker import ContentChunker
from core.parsers import ParserFactory, DocumentType


# Configure detailed logging - will be updated per document
def setup_logging(document_name: str = "test"):
    """Setup logging with document-specific log file"""
    # Ensure output directory exists
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with document name prefix
    log_file = output_dir / f"{document_name}_processing_log.txt"
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    
    return log_file

def configure_loggers():
    """Configure specific logger levels"""
    logging.getLogger('plugins.parsers.pdf_parser').setLevel(logging.INFO)
    logging.getLogger('core.content_chunker').setLevel(logging.INFO)
    logging.getLogger('core.chunking').setLevel(logging.INFO)
    logging.getLogger('core.clients.vllm_smoldocling_local').setLevel(logging.INFO)
    logging.getLogger('core.clients.vllm_qwen25_vl_local').setLevel(logging.INFO)
    logging.getLogger('core.vllm.base_client').setLevel(logging.INFO)
    logging.getLogger('core.vllm.model_manager').setLevel(logging.INFO)
    logging.getLogger('core.vllm_batch_processor').setLevel(logging.INFO)
    logging.getLogger('plugins.parsers.smoldocling_context_parser').setLevel(logging.INFO)
    # Enable detailed vLLM logging
    logging.getLogger('vllm').setLevel(logging.INFO)
    logging.getLogger('transformers').setLevel(logging.WARNING)  # Reduce transformers noise


async def test_document_processing(file_path: Path):
    """Test document processing with detailed logging"""
    
    # Setup logging for this specific document
    document_name = file_path.stem
    log_file = setup_logging(document_name)
    configure_loggers()
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING DOCUMENT PROCESSING TEST")
    logger.info(f"📄 File: {file_path.name}")
    logger.info(f"📏 Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"📝 Log file: {log_file}")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = get_config()
        
        # Phase 1: Document Parsing
        logger.info("\n" + "="*60)
        logger.info("📖 PHASE 1: DOCUMENT PARSING")
        logger.info("="*60)
        
        parser_factory = ParserFactory()
        parser = parser_factory.get_parser_for_file(file_path)
        
        if not parser:
            raise ValueError(f"No parser available for {file_path}")
            
        document = await parser.parse(file_path)
        
        logger.info(f"\n📊 PARSING RESULTS:")
        logger.info(f"  - Document Type: {document.metadata.document_type}")
        logger.info(f"  - Page Count: {document.metadata.page_count}")
        logger.info(f"  - Segments: {len(document.segments)}")
        logger.info(f"  - Visual Elements: {len(document.visual_elements)}")
        
        # Phase 2: Content Chunking
        logger.info("\n" + "="*60)
        logger.info("🔧 PHASE 2: CONTENT CHUNKING")
        logger.info("="*60)
        
        # Disable context inheritance for this test (no LLM calls)
        chunking_config = {
            "chunking": {
                "strategies": {
                    "pdf": {
                        "max_tokens": 500,
                        "min_tokens": 100,
                        "overlap_tokens": 50,
                        "respect_boundaries": True
                    }
                },
                "enable_context_inheritance": False,  # No LLM for this test
                "performance": {
                    "enable_async_processing": False,
                    "max_concurrent_groups": 1
                }
            }
        }
        
        chunker = ContentChunker(chunking_config)
        chunking_result = await chunker.chunk_document(document)
        
        logger.info(f"\n📊 CHUNKING RESULTS:")
        logger.info(f"  - Total Chunks: {chunking_result.total_chunks}")
        logger.info(f"  - Context Groups: {chunking_result.total_groups}")
        logger.info(f"  - Processing Time: {chunking_result.processing_stats.processing_time_seconds:.2f}s")
        
        # Display chunk details
        logger.info("\n📄 CHUNK DETAILS:")
        for i, chunk in enumerate(chunking_result.contextual_chunks[:5], 1):  # Show first 5
            logger.info(f"\n  Chunk {i}:")
            logger.info(f"    - ID: {chunk.chunk_id}")
            logger.info(f"    - Tokens: {chunk.token_count}")
            logger.info(f"    - Characters: {len(chunk.content)}")
            logger.info(f"    - Type: {chunk.chunk_type}")
            logger.info(f"    - Content Preview: {chunk.content[:100]}...")
        
        if len(chunking_result.contextual_chunks) > 5:
            logger.info(f"\n  ... and {len(chunking_result.contextual_chunks) - 5} more chunks")
        
        # Phase 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\n✅ Document processed successfully!")
        logger.info(f"  - Pages processed: {document.metadata.page_count}")
        logger.info(f"  - Segments created: {len(document.segments)}")
        logger.info(f"  - Chunks generated: {chunking_result.total_chunks}")
        logger.info(f"  - Average chunk size: {sum(c.token_count for c in chunking_result.contextual_chunks) / len(chunking_result.contextual_chunks):.0f} tokens")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        raise


async def test_simple_text():
    """Test with simple text document"""
    
    # Create test document
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "test_document.txt"
    test_file.write_text("""
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data, and makes predictions based on that data. Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning is used when the training data is not labeled. The system tries to learn without a teacher. Common algorithms include:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)

### Reinforcement Learning
In reinforcement learning, an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.

## Applications

Machine learning has wide applications across various domains:
1. Healthcare: Disease diagnosis and drug discovery
2. Finance: Credit scoring and fraud detection
3. Transportation: Autonomous vehicles
4. Entertainment: Recommendation systems

## Conclusion

Machine learning continues to evolve rapidly, with new techniques and applications emerging regularly. As computational power increases and more data becomes available, the potential for machine learning to solve complex problems continues to grow.
""")
    
    print("\n🧪 Testing with simple text document...")
    await test_document_processing(test_file)


async def test_pdf_with_vllm():
    """Test PDF processing with vLLM and detailed logging"""
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found at {pdf_path}")
        print("   To test with PDF, place your PDF file at that location")
        return
    
    print("\n\n" + "="*80)
    print("🧪 Testing with actual PDF document...")
    print("⚠️  NOTE: This WILL use vLLM models for SmolDocling PDF parsing")
    print("🔍 All vLLM activities will be captured in the log")
    print("="*80)
    
    # Test with PDF using vLLM
    await test_document_processing(pdf_path)


async def main():
    """Main test execution"""
    
    # Option 1: Test with simple text document
    await test_simple_text()
    
    # Option 2: Test with actual PDF using vLLM
    await test_pdf_with_vllm()
    
    print("\n✅ All tests completed! Check log files in /data/output/ for full details.")


if __name__ == "__main__":
    asyncio.run(main())