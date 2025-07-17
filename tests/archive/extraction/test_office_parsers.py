#!/usr/bin/env python3
"""
Test Office parsers (DOCX, XLSX, PPTX) with real files
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data/output/Office_Parsers_test_{timestamp}.log"

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
os.makedirs("data/output", exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Set up root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


async def test_docx_parser():
    """Test DOCX parser with Erkenntnisse_Herausforderungen.docx"""
    logger.info("\n" + "="*80)
    logger.info("📄 Testing DOCX Parser")
    logger.info("="*80)
    
    try:
        from core.parsers.implementations.office import DOCXParser
        
        docx_path = Path("data/input/Erkenntnisse_Herausforderungen.docx")
        if not docx_path.exists():
            logger.error(f"DOCX file not found: {docx_path}")
            return None
            
        logger.info(f"File: {docx_path.name}")
        logger.info(f"Size: {docx_path.stat().st_size / 1024:.2f} KB")
        
        # Create parser
        parser = DOCXParser(config={
            "extract_images": True,
            "extract_tables": True,
            "table_as_text": True
        }, enable_vlm=False)  # Disable VLM for now
        
        # Parse document
        start_time = datetime.now()
        document = await parser.parse(docx_path)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        # Log results
        logger.info(f"✅ DOCX parsing successful!")
        logger.info(f"⏱️ Parse time: {parse_time:.2f}s")
        logger.info(f"📊 Segments: {len(document.segments)}")
        logger.info(f"🖼️ Visual elements: {len(document.visual_elements)}")
        logger.info(f"📝 Title: {document.metadata.title}")
        logger.info(f"👤 Author: {document.metadata.author}")
        
        # Show segment types
        segment_types = {}
        for seg in document.segments:
            segment_types[seg.segment_type] = segment_types.get(seg.segment_type, 0) + 1
        logger.info(f"📑 Segment types: {segment_types}")
        
        # Show first few segments
        logger.info("\n--- First 3 segments ---")
        for i, seg in enumerate(document.segments[:3]):
            logger.info(f"Segment {i+1}: {seg.segment_type}")
            logger.info(f"Content preview: {seg.content[:100]}...")
            
        # Save to JSON
        output_file = f"data/output/DOCX_parse_result_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "file": str(docx_path),
                "parse_time": parse_time,
                "segments": len(document.segments),
                "visual_elements": len(document.visual_elements),
                "metadata": {
                    "title": document.metadata.title,
                    "author": document.metadata.author,
                    "created": str(document.metadata.created_date),
                    "page_count": document.metadata.page_count
                },
                "segment_types": segment_types
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to: {output_file}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ DOCX parsing failed: {e}", exc_info=True)
        return None


async def test_xlsx_parser():
    """Test XLSX parser with PapersWithCode_Abstracts.xlsx"""
    logger.info("\n" + "="*80)
    logger.info("📊 Testing XLSX Parser")
    logger.info("="*80)
    
    try:
        from core.parsers.implementations.office import XLSXParser
        
        xlsx_path = Path("data/input/PapersWithCode_Abstracts.xlsx")
        if not xlsx_path.exists():
            logger.error(f"XLSX file not found: {xlsx_path}")
            return None
            
        logger.info(f"File: {xlsx_path.name}")
        logger.info(f"Size: {xlsx_path.stat().st_size / 1024:.2f} KB")
        
        # Create parser
        parser = XLSXParser(config={
            "extract_charts": True,
            "extract_images": True,
            "extract_formulas": True,
            "sheet_as_segment": True
        }, enable_vlm=False)  # Disable VLM for now
        
        # Parse document
        start_time = datetime.now()
        document = await parser.parse(xlsx_path)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        # Log results
        logger.info(f"✅ XLSX parsing successful!")
        logger.info(f"⏱️ Parse time: {parse_time:.2f}s")
        logger.info(f"📊 Segments: {len(document.segments)}")
        logger.info(f"📈 Visual elements: {len(document.visual_elements)}")
        logger.info(f"📝 Title: {document.metadata.title}")
        
        # Show custom metadata
        if "sheets" in document.metadata.custom_metadata:
            logger.info(f"📋 Sheets: {document.metadata.custom_metadata['sheets']}")
        
        # Show segment info
        logger.info("\n--- Sheet segments ---")
        for i, seg in enumerate(document.segments[:5]):  # First 5 segments
            logger.info(f"Segment {i+1}: {seg.segment_type}")
            if "sheet_name" in seg.metadata:
                logger.info(f"  Sheet: {seg.metadata['sheet_name']}")
            logger.info(f"  Content preview: {seg.content[:100]}...")
            
        # Save to JSON
        output_file = f"data/output/XLSX_parse_result_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "file": str(xlsx_path),
                "parse_time": parse_time,
                "segments": len(document.segments),
                "visual_elements": len(document.visual_elements),
                "metadata": document.metadata.custom_metadata,
                "first_segment": document.segments[0].content[:200] if document.segments else ""
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to: {output_file}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ XLSX parsing failed: {e}", exc_info=True)
        return None


async def test_pptx_parser():
    """Test PPTX parser with CRISP-DM_TätigkeitenTools.pptx"""
    logger.info("\n" + "="*80)
    logger.info("🎯 Testing PPTX Parser")
    logger.info("="*80)
    
    try:
        from core.parsers.implementations.office import PPTXParser
        
        pptx_path = Path("data/input/CRISP-DM_TätigkeitenTools.pptx")
        if not pptx_path.exists():
            logger.error(f"PPTX file not found: {pptx_path}")
            return None
            
        logger.info(f"File: {pptx_path.name}")
        logger.info(f"Size: {pptx_path.stat().st_size / 1024:.2f} KB")
        
        # Create parser
        parser = PPTXParser(config={
            "extract_images": True,
            "extract_charts": True,
            "extract_diagrams": True,
            "extract_notes": True
        }, enable_vlm=False)  # Disable VLM for now
        
        # Parse document
        start_time = datetime.now()
        document = await parser.parse(pptx_path)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        # Log results
        logger.info(f"✅ PPTX parsing successful!")
        logger.info(f"⏱️ Parse time: {parse_time:.2f}s")
        logger.info(f"📊 Segments: {len(document.segments)}")
        logger.info(f"🖼️ Visual elements: {len(document.visual_elements)}")
        logger.info(f"📝 Title: {document.metadata.title}")
        logger.info(f"👤 Author: {document.metadata.author}")
        
        # Show custom metadata
        if "slides" in document.metadata.custom_metadata:
            logger.info(f"🎞️ Slides: {document.metadata.custom_metadata['slides']}")
        
        # Show segment types
        segment_types = {}
        for seg in document.segments:
            segment_types[seg.segment_type] = segment_types.get(seg.segment_type, 0) + 1
        logger.info(f"📑 Segment types: {segment_types}")
        
        # Show first few slides
        logger.info("\n--- First 3 slides ---")
        for i, seg in enumerate(document.segments[:3]):
            if seg.segment_type == "slide":
                logger.info(f"Slide {seg.metadata.get('slide_number', i+1)}: "
                          f"{seg.metadata.get('slide_title', 'No title')}")
                logger.info(f"  Content preview: {seg.content[:100]}...")
                
        # Show visual elements
        if document.visual_elements:
            logger.info(f"\n--- Visual elements ---")
            visual_types = {}
            for ve in document.visual_elements:
                visual_types[ve.element_type.value] = visual_types.get(ve.element_type.value, 0) + 1
            logger.info(f"Visual element types: {visual_types}")
            
        # Save to JSON
        output_file = f"data/output/PPTX_parse_result_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "file": str(pptx_path),
                "parse_time": parse_time,
                "segments": len(document.segments),
                "visual_elements": len(document.visual_elements),
                "metadata": {
                    "title": document.metadata.title,
                    "author": document.metadata.author,
                    "slides": document.metadata.custom_metadata.get("slides", 0)
                },
                "segment_types": segment_types,
                "visual_types": {ve.element_type.value: 
                               visual_types.get(ve.element_type.value, 0) 
                               for ve in document.visual_elements} if document.visual_elements else {}
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to: {output_file}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ PPTX parsing failed: {e}", exc_info=True)
        return None


async def main():
    """Run all office parser tests"""
    logger.info("="*80)
    logger.info("🏢 Starting Office Parser Tests")
    logger.info(f"📝 Log file: {log_filename}")
    logger.info("="*80)
    
    results = {
        "DOCX": None,
        "XLSX": None,
        "PPTX": None
    }
    
    # Test DOCX parser
    results["DOCX"] = await test_docx_parser()
    
    # Test XLSX parser
    results["XLSX"] = await test_xlsx_parser()
    
    # Test PPTX parser
    results["PPTX"] = await test_pptx_parser()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    
    success_count = sum(1 for r in results.values() if r is not None)
    logger.info(f"✅ Successful: {success_count}/3")
    logger.info(f"❌ Failed: {3 - success_count}/3")
    
    for format_name, result in results.items():
        if result:
            logger.info(f"✅ {format_name}: {len(result.segments)} segments, "
                       f"{len(result.visual_elements)} visual elements")
        else:
            logger.info(f"❌ {format_name}: Failed")
    
    logger.info(f"\n📄 Full log: {log_filename}")
    logger.info("✅ All tests completed!")
    
    return success_count == 3


if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n✅ All office parser tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)