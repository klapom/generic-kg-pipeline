#!/usr/bin/env python3
"""
Debug script to investigate bbox extraction issue
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tests/debugging/bbox_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Silence some noisy loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("pdf2image").setLevel(logging.WARNING)


async def debug_bbox_extraction():
    """Debug bbox extraction from SmolDocling"""
    logger.info("=" * 80)
    logger.info("BBOX EXTRACTION DEBUG")
    logger.info("=" * 80)
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    # Parse without VLM first
    logger.info("\n🔍 Parsing PDF with SmolDocling (no VLM)...")
    parser = HybridPDFParser(
        config={"pdfplumber_mode": 0},
        enable_vlm=False
    )
    
    document = await parser.parse(test_file)
    
    logger.info(f"\n📊 Document parsing complete:")
    logger.info(f"   - Total segments: {len(document.segments)}")
    logger.info(f"   - Total visual elements: {len(document.visual_elements)}")
    
    # Analyze visual elements
    logger.info("\n🖼️ Visual Elements Analysis:")
    
    elements_with_bbox = 0
    elements_without_bbox = 0
    
    for i, ve in enumerate(document.visual_elements[:10], 1):  # First 10 for debugging
        logger.info(f"\n--- Visual Element {i} ---")
        logger.info(f"   Type: {ve.element_type.value}")
        logger.info(f"   Page: {ve.page_or_slide}")
        logger.info(f"   Hash: {ve.content_hash[:16]}...")
        
        # Check bounding box
        if ve.bounding_box:
            elements_with_bbox += 1
            logger.info(f"   ✅ Bounding Box: {ve.bounding_box}")
        else:
            elements_without_bbox += 1
            logger.info(f"   ❌ Bounding Box: None")
        
        # Check analysis metadata
        if ve.analysis_metadata:
            logger.info(f"   Metadata keys: {list(ve.analysis_metadata.keys())}")
            if 'raw_bbox' in ve.analysis_metadata:
                logger.info(f"   Raw bbox: {ve.analysis_metadata['raw_bbox']}")
            if 'caption' in ve.analysis_metadata:
                logger.info(f"   Caption: {ve.analysis_metadata['caption']}")
        else:
            logger.info(f"   Metadata: None")
    
    logger.info(f"\n📈 Summary:")
    logger.info(f"   - Elements with bbox: {elements_with_bbox}")
    logger.info(f"   - Elements without bbox: {elements_without_bbox}")
    
    # Check if visual elements are linked to segments
    logger.info("\n🔗 Visual Element to Segment Mapping:")
    
    for i, segment in enumerate(document.segments):
        if segment.segment_type == "visual" and segment.visual_references:
            logger.info(f"\n   Segment {i}: {segment.segment_type}/{segment.segment_subtype}")
            logger.info(f"   Page: {segment.page_number}")
            logger.info(f"   Content: {segment.content[:50]}...")
            logger.info(f"   Visual refs: {len(segment.visual_references)} references")
            
            # Find corresponding visual element
            for ref_hash in segment.visual_references[:1]:  # First reference only
                matching_ve = next((ve for ve in document.visual_elements if ve.content_hash == ref_hash), None)
                if matching_ve:
                    logger.info(f"   → Matched VE: bbox={matching_ve.bounding_box is not None}")
                else:
                    logger.info(f"   → No matching VE found for hash {ref_hash[:16]}...")


if __name__ == "__main__":
    asyncio.run(debug_bbox_extraction())