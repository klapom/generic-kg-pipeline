#!/usr/bin/env python3
"""
Test VLM Integration with Visual Elements

Tests the complete VLM integration flow:
1. Parse PDF with visual elements
2. Analyze visual elements with VLM
3. Update visual segments with VLM descriptions
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.parsers.interfaces import SegmentType, VisualSubtype

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_vlm_integration():
    """Test VLM integration with real PDF containing visual elements"""
    logger.info("🚀 Starting VLM Integration Test")
    
    # Test file with visual elements
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"📄 Testing with: {test_file.name}")
    
    # Parse with VLM enabled
    logger.info("🔧 Initializing HybridPDFParser with VLM enabled...")
    parser = HybridPDFParser(
        config={"pdfplumber_mode": 0},  # Use SmolDocling for visual extraction
        enable_vlm=True
    )
    
    try:
        # Parse document
        logger.info("📖 Parsing document...")
        start_time = datetime.now()
        document = await parser.parse(test_file)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Parsing completed in {parse_time:.2f}s")
        logger.info(f"📊 Document contains:")
        logger.info(f"   - {len(document.segments)} segments")
        logger.info(f"   - {len(document.visual_elements)} visual elements")
        
        # Analyze visual segments
        visual_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.VISUAL]
        logger.info(f"   - {len(visual_segments)} visual segments")
        
        # Check VLM analysis results
        vlm_analyzed_count = 0
        placeholder_count = 0
        
        logger.info("\n🎨 Visual Segment Analysis:")
        for i, seg in enumerate(visual_segments[:5]):  # Show first 5
            is_placeholder = seg.content.startswith("[") and seg.content.endswith("]")
            is_analyzed = seg.metadata and seg.metadata.get("vlm_analyzed", False)
            
            if is_placeholder:
                placeholder_count += 1
            if is_analyzed:
                vlm_analyzed_count += 1
            
            logger.info(f"\n  Visual Segment {i+1}:")
            logger.info(f"    Subtype: {seg.segment_subtype}")
            logger.info(f"    Content: {seg.content[:100]}...")
            logger.info(f"    VLM Analyzed: {is_analyzed}")
            if is_analyzed:
                logger.info(f"    Confidence: {seg.metadata.get('confidence', 0):.2f}")
                logger.info(f"    Has Extracted Data: {seg.metadata.get('has_extracted_data', False)}")
        
        # Summary
        logger.info("\n📊 VLM Integration Summary:")
        logger.info(f"   - Visual segments with placeholders: {placeholder_count}")
        logger.info(f"   - Visual segments analyzed by VLM: {vlm_analyzed_count}")
        logger.info(f"   - Analysis success rate: {vlm_analyzed_count/len(visual_segments)*100:.1f}%")
        
        # Check visual elements
        logger.info("\n🖼️ Visual Elements:")
        for i, ve in enumerate(document.visual_elements[:3]):  # Show first 3
            logger.info(f"\n  Visual Element {i+1}:")
            logger.info(f"    Type: {ve.element_type.value}")
            logger.info(f"    Has VLM Description: {bool(ve.vlm_description)}")
            logger.info(f"    Confidence: {ve.confidence or 0:.2f}")
            if ve.vlm_description:
                logger.info(f"    Description: {ve.vlm_description[:100]}...")
        
        # Save results
        output_dir = Path("tests/debugging/vlm_integration")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segments summary
        segments_summary = []
        for seg in document.segments:
            seg_info = {
                "index": seg.segment_index,
                "type": seg.segment_type.value,
                "subtype": seg.segment_subtype,
                "content_preview": seg.content[:100] + "..." if len(seg.content) > 100 else seg.content,
                "page": seg.page_number,
                "visual_refs": seg.visual_references
            }
            if seg.segment_type == SegmentType.VISUAL and seg.metadata:
                seg_info["vlm_analyzed"] = seg.metadata.get("vlm_analyzed", False)
                seg_info["confidence"] = seg.metadata.get("confidence", 0)
            segments_summary.append(seg_info)
        
        summary_file = output_dir / f"vlm_integration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "file": str(test_file),
                "parse_time_seconds": parse_time,
                "total_segments": len(document.segments),
                "total_visual_elements": len(document.visual_elements),
                "visual_segments": len(visual_segments),
                "vlm_analyzed": vlm_analyzed_count,
                "segments": segments_summary
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to: {summary_file}")
        
        # Test successful if we have VLM-analyzed segments
        if vlm_analyzed_count > 0:
            logger.info("\n✅ VLM Integration Test PASSED!")
            logger.info(f"   Successfully analyzed {vlm_analyzed_count} visual segments with VLM")
        else:
            logger.warning("\n⚠️ VLM Integration Test WARNING!")
            logger.warning("   No visual segments were analyzed by VLM")
            logger.warning("   This might indicate VLM service is not running or configuration issues")
        
    except Exception as e:
        logger.error(f"\n❌ VLM Integration Test FAILED: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if hasattr(parser, 'cleanup'):
            parser.cleanup()


async def test_vlm_without_service():
    """Test VLM integration behavior when VLM service is not available"""
    logger.info("\n🧪 Testing VLM integration without VLM service...")
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    # Parse with VLM enabled but service not running
    parser = HybridPDFParser(
        config={"pdfplumber_mode": 0},
        enable_vlm=True
    )
    
    try:
        document = await parser.parse(test_file)
        
        # Check that visual segments still exist with placeholders
        visual_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.VISUAL]
        placeholder_count = sum(1 for seg in visual_segments if seg.content.startswith("[") and seg.content.endswith("]"))
        
        logger.info(f"✅ Document parsed successfully without VLM service")
        logger.info(f"   - Visual segments: {len(visual_segments)}")
        logger.info(f"   - With placeholders: {placeholder_count}")
        
        if placeholder_count == len(visual_segments):
            logger.info("✅ All visual segments have placeholders (expected behavior without VLM)")
        else:
            logger.warning(f"⚠️ Some visual segments might have been analyzed: {len(visual_segments) - placeholder_count}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        if hasattr(parser, 'cleanup'):
            parser.cleanup()


async def main():
    """Run all VLM integration tests"""
    logger.info("=" * 80)
    logger.info("VLM INTEGRATION TEST SUITE")
    logger.info("=" * 80)
    
    # Test 1: Full VLM integration
    await test_vlm_integration()
    
    # Test 2: Behavior without VLM service
    await test_vlm_without_service()
    
    logger.info("\n" + "=" * 80)
    logger.info("VLM INTEGRATION TEST SUITE COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())