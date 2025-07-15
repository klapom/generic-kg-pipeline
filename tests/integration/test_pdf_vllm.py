#!/usr/bin/env python3
"""
Test script specifically for PDF processing with vLLM
Focus on detailed logging of vLLM activities
"""

import logging
import sys
import asyncio
import os
from pathlib import Path
from datetime import datetime

# Enable vLLM debug logging
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import get_config
from core.content_chunker import ContentChunker
from plugins.parsers.parser_factory import ParserFactory
from plugins.parsers.base_parser import DocumentType


def setup_pdf_logging(pdf_name: str):
    """Setup comprehensive logging for PDF processing"""
    # Ensure output directory exists
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with PDF name prefix
    log_file = output_dir / f"{pdf_name}_vllm_processing_log.txt"
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    
    # Configure all relevant loggers for vLLM visibility
    loggers_to_configure = [
        'plugins.parsers.pdf_parser',
        'core.content_chunker',
        'core.chunking',
        'core.clients.vllm_smoldocling_local',
        'core.clients.vllm_qwen25_vl_local',
        'core.vllm.base_client',
        'core.vllm.model_manager',
        'core.vllm_batch_processor',
        'plugins.parsers.smoldocling_context_parser',
        'vllm',
        'core.clients.vllm_smoldocling'
    ]
    
    for logger_name in loggers_to_configure:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Reduce noise from other libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return log_file


async def test_pdf_processing_with_vllm(pdf_path: Path):
    """Test PDF processing with comprehensive vLLM logging"""
    
    # Setup logging for this specific PDF
    pdf_name = pdf_path.stem
    log_file = setup_pdf_logging(pdf_name)
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING PDF PROCESSING WITH vLLM")
    logger.info(f"📄 PDF File: {pdf_path.name}")
    logger.info(f"📏 Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"📝 Log file: {log_file}")
    logger.info("🔧 vLLM Models: SmolDocling for PDF parsing")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = get_config()
        
        # Phase 1: PDF Parsing with vLLM
        logger.info("\n" + "="*60)
        logger.info("📖 PHASE 1: PDF PARSING WITH vLLM")
        logger.info("="*60)
        
        # Create parser factory with VLM enabled
        parser_factory = ParserFactory(enable_vlm=True)
        parser = parser_factory.get_parser_for_file(pdf_path)
        
        if not parser:
            raise ValueError(f"No parser available for {pdf_path}")
        
        logger.info(f"🔍 Using parser: {type(parser).__name__}")
        logger.info("🚀 Starting vLLM-based PDF parsing...")
        
        # Parse with vLLM
        document = await parser.parse(pdf_path)
        
        logger.info(f"\n📊 PDF PARSING RESULTS:")
        logger.info(f"  - Document Type: {document.metadata.document_type}")
        logger.info(f"  - Page Count: {document.metadata.page_count}")
        logger.info(f"  - Segments: {len(document.segments)}")
        logger.info(f"  - Visual Elements: {len(document.visual_elements)}")
        logger.info(f"  - Title: {document.metadata.title}")
        
        # Show first few segments
        logger.info(f"\n📄 FIRST 5 SEGMENTS:")
        for i, segment in enumerate(document.segments[:5], 1):
            logger.info(f"  Segment {i}: {segment.segment_type}, {len(segment.content)} chars")
            logger.info(f"    Preview: {segment.content[:100]}...")
        
        # Phase 2: Content Chunking
        logger.info("\n" + "="*60)
        logger.info("🔧 PHASE 2: CONTENT CHUNKING")
        logger.info("="*60)
        
        # Configure chunking for PDF
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
                "enable_context_inheritance": False,  # Disable LLM calls for now
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
        for i, chunk in enumerate(chunking_result.contextual_chunks[:3], 1):  # Show first 3
            logger.info(f"\n  Chunk {i}:")
            logger.info(f"    - ID: {chunk.chunk_id}")
            logger.info(f"    - Tokens: {chunk.token_count}")
            logger.info(f"    - Characters: {len(chunk.content)}")
            logger.info(f"    - Type: {chunk.chunk_type}")
            logger.info(f"    - Content Preview: {chunk.content[:150]}...")
        
        if len(chunking_result.contextual_chunks) > 3:
            logger.info(f"\n  ... and {len(chunking_result.contextual_chunks) - 3} more chunks")
        
        # Phase 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 PROCESSING SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\n✅ PDF processed successfully with vLLM!")
        logger.info(f"  - Pages processed: {document.metadata.page_count}")
        logger.info(f"  - Segments created: {len(document.segments)}")
        logger.info(f"  - Visual elements: {len(document.visual_elements)}")
        logger.info(f"  - Chunks generated: {chunking_result.total_chunks}")
        logger.info(f"  - Average chunk size: {sum(c.token_count for c in chunking_result.contextual_chunks) / len(chunking_result.contextual_chunks):.0f} tokens")
        
        return document, chunking_result
        
    except Exception as e:
        logger.error(f"\n❌ ERROR during PDF processing: {str(e)}", exc_info=True)
        raise


async def main():
    """Main test execution for PDF with vLLM"""
    
    # Test with BMW PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found at {pdf_path}")
        print("   Available PDFs:")
        for pdf_file in Path("data/input").glob("*.pdf"):
            print(f"   - {pdf_file.name}")
        return
    
    print(f"\n🚀 Starting PDF processing with vLLM")
    print(f"📄 Processing: {pdf_path.name}")
    print(f"📏 Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        document, chunking_result = await test_pdf_processing_with_vllm(pdf_path)
        
        print(f"\n✅ PDF processing completed successfully!")
        print(f"📊 Results: {len(document.segments)} segments, {chunking_result.total_chunks} chunks")
        print(f"📝 Check log file: data/output/{pdf_path.stem}_vllm_processing_log.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"📝 Check log file for details: data/output/{pdf_path.stem}_vllm_processing_log.txt")


if __name__ == "__main__":
    asyncio.run(main())