#!/usr/bin/env python3
"""
Debug SmolDocling image extraction
"""

import asyncio
from pathlib import Path
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def debug_images():
    """Debug why SmolDocling isn't finding images"""
    
    # Test with BMW document
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    parser = HybridPDFParserQwen25(
        config={
            "max_pages": 3,
            "pdfplumber_mode": 1
        },
        enable_vlm=False
    )
    
    logger.info(f"Parsing {pdf_path.name}...")
    document = await parser.parse(pdf_path)
    
    logger.info(f"\nDocument stats:")
    logger.info(f"  Segments: {len(document.segments)}")
    logger.info(f"  Visual elements: {len(document.visual_elements)}")
    
    # Check segments for image tags
    logger.info("\nChecking segments for image tags...")
    image_pattern = r'<image>.*?</image>'
    figure_pattern = r'<figure>.*?</figure>'
    loc_pattern = r'<loc_\d+>'
    
    for i, segment in enumerate(document.segments):
        # Look for image tags
        images = re.findall(image_pattern, segment.content)
        figures = re.findall(figure_pattern, segment.content)
        locs = re.findall(loc_pattern, segment.content)
        
        if images or figures:
            logger.info(f"\nSegment {i} (page {segment.page_number}):")
            logger.info(f"  Found {len(images)} image tags")
            logger.info(f"  Found {len(figures)} figure tags")
            
            # Show first few matches
            for img in images[:3]:
                logger.info(f"    Image: {img[:100]}...")
            for fig in figures[:3]:
                logger.info(f"    Figure: {fig[:100]}...")
        
        # Show segments with many loc tags (might be tables or images)
        if len(locs) > 20:
            logger.info(f"\nSegment {i} has {len(locs)} loc tags - might contain visual elements")
            logger.info(f"  Content preview: {segment.content[:200]}...")
    
    # Check raw SmolDocling output
    logger.info("\nChecking SmolDocling client directly...")
    result = parser.smoldocling_client.parse_pdf(pdf_path)
    
    logger.info(f"SmolDocling result:")
    logger.info(f"  Pages: {len(result.pages)}")
    logger.info(f"  Visual elements: {len(result.visual_elements)}")
    
    for i, page in enumerate(result.pages[:3]):
        logger.info(f"\nPage {i+1}:")
        logger.info(f"  Tables: {len(page.tables)}")
        logger.info(f"  Images: {len(page.images)}")
        logger.info(f"  Formulas: {len(page.formulas)}")
        if hasattr(page, 'visual_elements'):
            logger.info(f"  Visual elements: {len(page.visual_elements)}")
    
    parser.cleanup()


if __name__ == "__main__":
    asyncio.run(debug_images())