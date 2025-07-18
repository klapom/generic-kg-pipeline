#!/usr/bin/env python3
"""
Test Qwen2.5-VL Integration

This test verifies the new single-stage Qwen2.5-VL processor integration
with the enhanced HybridPDFParser.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25
from core.parsers.interfaces import SegmentType, VisualElementType

# Setup logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'qwen25_integration_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_qwen25_integration():
    """Test the complete Qwen2.5-VL integration"""
    
    logger.info("=" * 80)
    logger.info("🚀 QWEN2.5-VL INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Test files
    test_files = [
        Path("data/input/Preview_BMW_3er_G20.pdf"),
        Path("data/input/test_simple.pdf")
    ]
    
    results = {}
    
    for test_file in test_files:
        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}")
            continue
        
        logger.info(f"\n📄 Testing: {test_file.name}")
        logger.info("-" * 60)
        
        try:
            # Test 1: Without VLM
            logger.info("\n🔍 TEST 1: Parsing without VLM...")
            start_time = time.time()
            
            parser_no_vlm = HybridPDFParserQwen25(
                config={
                    "pdfplumber_mode": 1,
                    "max_pages": 10,
                    "enable_page_context": False
                },
                enable_vlm=False
            )
            
            doc_no_vlm = await parser_no_vlm.parse(test_file)
            time_no_vlm = time.time() - start_time
            
            logger.info(f"✅ Completed in {time_no_vlm:.2f}s")
            logger.info(f"   Segments: {len(doc_no_vlm.segments)}")
            logger.info(f"   Visual elements: {len(doc_no_vlm.visual_elements)}")
            
            # Test 2: With VLM but no page context
            logger.info("\n🧠 TEST 2: Parsing with Qwen2.5-VL (no page context)...")
            start_time = time.time()
            
            parser_vlm_basic = HybridPDFParserQwen25(
                config={
                    "pdfplumber_mode": 1,
                    "max_pages": 5,  # Limit for testing
                    "enable_page_context": False,
                    "vlm": {
                        "temperature": 0.2,
                        "max_new_tokens": 512,
                        "batch_size": 2,
                        "enable_structured_parsing": True
                    }
                },
                enable_vlm=True
            )
            
            doc_vlm_basic = await parser_vlm_basic.parse(test_file)
            time_vlm_basic = time.time() - start_time
            
            logger.info(f"✅ Completed in {time_vlm_basic:.2f}s")
            
            # Count VLM analyzed elements
            vlm_analyzed = sum(1 for ve in doc_vlm_basic.visual_elements if ve.vlm_description)
            logger.info(f"   VLM analyzed: {vlm_analyzed}/{len(doc_vlm_basic.visual_elements)}")
            
            # Test 3: With VLM and page context
            logger.info("\n🌐 TEST 3: Parsing with Qwen2.5-VL + page context...")
            start_time = time.time()
            
            parser_vlm_full = HybridPDFParserQwen25(
                config={
                    "pdfplumber_mode": 1,
                    "max_pages": 3,  # Even more limited for context analysis
                    "enable_page_context": True,
                    "page_context_pages": 3,
                    "vlm": {
                        "temperature": 0.2,
                        "max_new_tokens": 512,
                        "batch_size": 2,
                        "enable_structured_parsing": True
                    },
                    "image_extraction": {
                        "min_size": 100,
                        "extract_embedded": True,
                        "render_fallback": True,
                        "page_render_dpi": 150
                    }
                },
                enable_vlm=True
            )
            
            doc_vlm_full = await parser_vlm_full.parse(test_file)
            time_vlm_full = time.time() - start_time
            
            logger.info(f"✅ Completed in {time_vlm_full:.2f}s")
            
            # Analyze results
            vlm_analyzed_full = sum(1 for ve in doc_vlm_full.visual_elements if ve.vlm_description)
            with_context = sum(1 for ve in doc_vlm_full.visual_elements 
                             if hasattr(ve, 'analysis_metadata') and 
                             ve.analysis_metadata.get('page_context'))
            with_structured = sum(1 for ve in doc_vlm_full.visual_elements
                                if hasattr(ve, 'analysis_metadata') and
                                ve.analysis_metadata.get('structured_data'))
            
            logger.info(f"   VLM analyzed: {vlm_analyzed_full}/{len(doc_vlm_full.visual_elements)}")
            logger.info(f"   With page context: {with_context}")
            logger.info(f"   With structured data: {with_structured}")
            
            # Detailed analysis of visual elements
            logger.info("\n📊 VISUAL ELEMENT ANALYSIS:")
            logger.info("-" * 40)
            
            for i, ve in enumerate(doc_vlm_full.visual_elements[:5]):  # First 5
                logger.info(f"\n🖼️ Visual Element {i+1}:")
                logger.info(f"   Type: {ve.element_type.value}")
                logger.info(f"   Page: {ve.page_or_slide}")
                logger.info(f"   Has raw data: {bool(ve.raw_data)}")
                logger.info(f"   Has VLM description: {bool(ve.vlm_description)}")
                
                if ve.vlm_description:
                    logger.info(f"   Description preview: {ve.vlm_description[:100]}...")
                
                if hasattr(ve, 'analysis_metadata'):
                    if ve.analysis_metadata.get('structured_data'):
                        struct_type = ve.analysis_metadata['structured_data'].get('type')
                        logger.info(f"   Structured data type: {struct_type}")
                    
                    if ve.analysis_metadata.get('page_context'):
                        context = ve.analysis_metadata['page_context']
                        logger.info(f"   Page type: {context.get('type')}")
                        logger.info(f"   Main topic: {context.get('main_topic')}")
            
            # Store results
            results[test_file.name] = {
                "without_vlm": {
                    "time": time_no_vlm,
                    "segments": len(doc_no_vlm.segments),
                    "visual_elements": len(doc_no_vlm.visual_elements)
                },
                "with_vlm_basic": {
                    "time": time_vlm_basic,
                    "vlm_analyzed": vlm_analyzed
                },
                "with_vlm_full": {
                    "time": time_vlm_full,
                    "vlm_analyzed": vlm_analyzed_full,
                    "with_context": with_context,
                    "with_structured": with_structured
                }
            }
            
            # Cleanup
            parser_no_vlm.cleanup()
            parser_vlm_basic.cleanup()
            parser_vlm_full.cleanup()
            
        except Exception as e:
            logger.error(f"Error testing {test_file}: {e}", exc_info=True)
            results[test_file.name] = {"error": str(e)}
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    for filename, result in results.items():
        logger.info(f"\n📄 {filename}:")
        if "error" in result:
            logger.info(f"   ❌ Error: {result['error']}")
        else:
            logger.info(f"   Without VLM: {result['without_vlm']['time']:.2f}s")
            logger.info(f"   With VLM (basic): {result['with_vlm_basic']['time']:.2f}s")
            logger.info(f"   With VLM (full): {result['with_vlm_full']['time']:.2f}s")
            logger.info(f"   VLM analyzed: {result['with_vlm_full']['vlm_analyzed']}")
            logger.info(f"   Structured data: {result['with_vlm_full']['with_structured']}")
    
    # Save results
    output_file = log_dir / f"qwen25_integration_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    return results


async def test_structured_json_parsing():
    """Test structured JSON parsing capabilities"""
    
    logger.info("\n" + "=" * 80)
    logger.info("🔍 STRUCTURED JSON PARSING TEST")
    logger.info("=" * 80)
    
    # This would test the JSON parsing capabilities specifically
    # For now, included in main test above
    pass


async def test_page_context_analysis():
    """Test page-level context analysis"""
    
    logger.info("\n" + "=" * 80)
    logger.info("📄 PAGE CONTEXT ANALYSIS TEST")
    logger.info("=" * 80)
    
    # This would test page context analysis specifically
    # For now, included in main test above
    pass


if __name__ == "__main__":
    # Run all tests
    results = asyncio.run(test_qwen25_integration())
    
    # Additional specific tests if needed
    # asyncio.run(test_structured_json_parsing())
    # asyncio.run(test_page_context_analysis())
    
    if results:
        print("\n✅ Qwen2.5-VL integration tests completed!")
    else:
        print("\n❌ Tests failed!")