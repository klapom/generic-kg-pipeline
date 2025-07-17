#!/usr/bin/env python3
"""
Test BMW Document with SmolDocling
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.parsers.interfaces import SegmentType

# Configure logging to both file and console
log_file = Path("tests/debugging/vlm_integration/bmw_smoldocling_test.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_bmw_with_smoldocling():
    """Test BMW document with SmolDocling (no VLM)"""
    logger.info("=" * 80)
    logger.info("BMW Document SmolDocling Test")
    logger.info("=" * 80)
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"📄 Processing: {test_file.name}")
    logger.info("🤖 Parser: SmolDocling (vLLM)")
    logger.info("🔧 VLM: DISABLED")
    
    # Initialize parser
    parser = HybridPDFParser(
        config={
            "pdfplumber_mode": 0,  # Use only SmolDocling
            "log_level": "INFO"
        },
        enable_vlm=False
    )
    
    try:
        logger.info("\n📖 Starting document parsing...")
        start_time = datetime.now()
        
        document = await parser.parse(test_file)
        
        parse_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Parsing completed in {parse_time:.2f} seconds")
        
        # Document statistics
        logger.info(f"\n📊 Document Statistics:")
        logger.info(f"   - Pages: {document.metadata.page_count}")
        logger.info(f"   - Total segments: {len(document.segments)}")
        logger.info(f"   - Visual elements: {len(document.visual_elements)}")
        
        # Segment distribution
        segment_stats = {}
        for seg in document.segments:
            key = f"{seg.segment_type.value}/{seg.segment_subtype or 'none'}"
            segment_stats[key] = segment_stats.get(key, 0) + 1
        
        logger.info("\n📊 Segment Distribution:")
        for key, count in sorted(segment_stats.items()):
            logger.info(f"   - {key}: {count}")
        
        # Visual segments
        visual_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.VISUAL]
        logger.info(f"\n🎨 Visual Segments: {len(visual_segments)}")
        
        # Show sample segments
        logger.info("\n📝 Sample Segments:")
        
        # Text segments
        text_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.TEXT][:3]
        for i, seg in enumerate(text_segments):
            logger.info(f"\n  Text Segment {i+1} (Page {seg.page_number}):")
            logger.info(f"    Subtype: {seg.segment_subtype}")
            content_preview = seg.content[:100] + "..." if len(seg.content) > 100 else seg.content
            logger.info(f"    Content: {content_preview}")
        
        # Visual segments
        for i, seg in enumerate(visual_segments[:5]):
            logger.info(f"\n  Visual Segment {i+1} (Page {seg.page_number}):")
            logger.info(f"    Subtype: {seg.segment_subtype}")
            logger.info(f"    Content: {seg.content}")
            logger.info(f"    Visual refs: {seg.visual_references}")
        
        # Visual elements
        logger.info(f"\n🖼️ Visual Elements: {len(document.visual_elements)}")
        for i, ve in enumerate(document.visual_elements[:5]):
            logger.info(f"\n  Visual Element {i+1}:")
            logger.info(f"    Type: {ve.element_type.value}")
            logger.info(f"    Page: {ve.page_or_slide}")
            logger.info(f"    Hash: {ve.content_hash[:16]}...")
        
        # Save results
        output_dir = Path("tests/debugging/vlm_integration")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive output
        output_data = {
            "file": str(test_file),
            "timestamp": datetime.now().isoformat(),
            "parse_time_seconds": parse_time,
            "document_metadata": {
                "title": document.metadata.title,
                "page_count": document.metadata.page_count,
                "file_size": document.metadata.file_size
            },
            "statistics": {
                "total_segments": len(document.segments),
                "visual_elements": len(document.visual_elements),
                "visual_segments": len(visual_segments),
                "segment_distribution": segment_stats
            },
            "segments": []
        }
        
        # Add all segments
        for seg in document.segments:
            seg_data = {
                "index": seg.segment_index,
                "type": seg.segment_type.value,
                "subtype": seg.segment_subtype,
                "page": seg.page_number,
                "content": seg.content,
                "visual_references": seg.visual_references,
                "metadata": seg.metadata
            }
            output_data["segments"].append(seg_data)
        
        # Add visual elements
        output_data["visual_elements"] = []
        for ve in document.visual_elements:
            ve_data = {
                "element_type": ve.element_type.value,
                "page": ve.page_or_slide,
                "content_hash": ve.content_hash,
                "bounding_box": ve.bounding_box
            }
            output_data["visual_elements"].append(ve_data)
        
        # Save JSON
        json_file = output_dir / f"bmw_smoldocling_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to:")
        logger.info(f"   - JSON: {json_file}")
        logger.info(f"   - Log: {log_file}")
        
        logger.info("\n✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise
    finally:
        if hasattr(parser, 'cleanup'):
            parser.cleanup()
        logger.info("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_bmw_with_smoldocling())