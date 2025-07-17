#!/usr/bin/env python3
"""Quick test for Office parsers with new segment structure"""

import asyncio
import logging
from pathlib import Path

from core.parsers.interfaces import SegmentType, MetadataSubtype, TableSubtype, TextSubtype
from core.parsers.implementations.office import DOCXParser, XLSXParser, PPTXParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_docx():
    """Test DOCX parser"""
    logger.info("\n=== DOCX Parser Test ===")
    
    test_file = Path("data/input/Erkenntnisse_Herausforderungen.docx")
    parser = DOCXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    logger.info(f"✅ Parsed {len(document.segments)} segments")
    
    # Sample first few segments
    for i, seg in enumerate(document.segments[:3]):
        logger.info(f"  Segment {i+1}: {seg.segment_type.value}/{seg.segment_subtype} - {seg.content[:40]}...")
    
    return document


async def test_xlsx():
    """Test XLSX parser"""
    logger.info("\n=== XLSX Parser Test ===")
    
    test_file = Path("data/input/PapersWithCode_Abstracts.xlsx")
    parser = XLSXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    logger.info(f"✅ Parsed {len(document.segments)} segments")
    
    # Check for proper types
    metadata_count = sum(1 for seg in document.segments if seg.segment_type == SegmentType.METADATA)
    table_count = sum(1 for seg in document.segments if seg.segment_type == SegmentType.TABLE)
    
    logger.info(f"  Metadata segments: {metadata_count}")
    logger.info(f"  Table segments: {table_count}")
    
    # Sample segments
    for seg in document.segments[:3]:
        logger.info(f"  {seg.segment_type.value}/{seg.segment_subtype}: {seg.content[:50]}...")
    
    return document


async def test_pptx():
    """Test PPTX parser"""
    logger.info("\n=== PPTX Parser Test ===")
    
    test_file = Path("data/input/CRISP-DM_TätigkeitenTools.pptx")
    parser = PPTXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    logger.info(f"✅ Parsed {len(document.segments)} segments")
    
    # Check slide metadata
    slide_count = sum(1 for seg in document.segments 
                     if seg.segment_type == SegmentType.METADATA 
                     and seg.segment_subtype == MetadataSubtype.SLIDE.value)
    
    logger.info(f"  Slide segments: {slide_count}")
    logger.info(f"  Visual elements: {len(document.visual_elements)}")
    
    # Sample slides
    for seg in document.segments[:5]:
        if seg.segment_type == SegmentType.METADATA:
            logger.info(f"  {seg.metadata.get('slide_title', 'No title')}")
    
    return document


async def main():
    """Run all tests"""
    logger.info("🚀 Testing Office parsers with new segment structure")
    
    docx_doc = await test_docx()
    xlsx_doc = await test_xlsx()  
    pptx_doc = await test_pptx()
    
    logger.info("\n✅ All Office parsers working with new segment structure!")
    
    # Summary stats
    logger.info("\n📊 Summary:")
    logger.info(f"  DOCX: {len(docx_doc.segments)} segments")
    logger.info(f"  XLSX: {len(xlsx_doc.segments)} segments")
    logger.info(f"  PPTX: {len(pptx_doc.segments)} segments, {len(pptx_doc.visual_elements)} visuals")


if __name__ == "__main__":
    asyncio.run(main())