#!/usr/bin/env python3
"""
Test production pipeline with Qwen2.5-VL for image analysis only
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.parsers.parser_factory import ParserFactory

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_test():
    """Test the production pipeline with BMW document"""
    
    # Select BMW document
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info(f"🚀 Testing production pipeline with: {pdf_path.name}")
    
    # Configure parser factory - Standard HybridPDFParser with Qwen2.5-VL for images
    parser_config = {
        'max_pages': 5,  # Limit for testing
        'pdfplumber_mode': 1,  # Fallback mode
        'enable_vlm': True,  # Enable VLM for image analysis
        'environment': 'production',
        'extract_images': True,
        'extract_tables': True,
        'image_min_size': 100
    }
    
    # Create parser factory
    factory = ParserFactory(config=parser_config, enable_vlm=True)
    
    # Get parser (should be HybridPDFParser)
    parser = factory.get_parser_for_file(pdf_path)
    logger.info(f"Using parser: {parser.__class__.__name__}")
    
    # Parse document
    logger.info("📄 Parsing document...")
    document = await parser.parse(pdf_path)
    
    logger.info(f"✅ Parsing completed:")
    logger.info(f"   - Pages: {document.metadata.page_count}")
    logger.info(f"   - Segments: {len(document.segments)}")
    logger.info(f"   - Visual elements: {len(document.visual_elements)}")
    
    # Check visual elements with VLM analysis
    vlm_analyzed = 0
    for ve in document.visual_elements:
        if ve.vlm_description:
            vlm_analyzed += 1
            logger.info(f"   - {ve.element_type.value}: {ve.vlm_description[:100]}...")
    
    logger.info(f"   - VLM analyzed: {vlm_analyzed}/{len(document.visual_elements)}")
    
    # Count segment types
    segment_types = {}
    for segment in document.segments:
        seg_type = f"{segment.segment_type.value}/{segment.segment_subtype or 'none'}"
        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
    
    logger.info("📊 Segment types:")
    for seg_type, count in segment_types.items():
        logger.info(f"   - {seg_type}: {count}")
    
    # Check for tables
    tables = [s for s in document.segments if s.segment_type.value == 'table']
    if tables:
        logger.info(f"📊 Found {len(tables)} tables")
    
    # Cleanup
    if hasattr(parser, 'cleanup'):
        parser.cleanup()
    
    print(f"\n✅ Test completed successfully!")
    print(f"📄 Document: {pdf_path.name}")
    print(f"🖼️ Visual elements analyzed with Qwen2.5-VL: {vlm_analyzed}")


if __name__ == "__main__":
    asyncio.run(run_test())