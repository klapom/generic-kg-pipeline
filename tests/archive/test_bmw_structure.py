#!/usr/bin/env python3
"""
Test BMW Document Structure ohne VLM
Testet die neue Segment-Struktur mit dem BMW Dokument
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_bmw_structure():
    """Test BMW document structure without VLM"""
    logger.info("🚀 Starting BMW Document Structure Test")
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"📄 Processing: {test_file.name}")
    logger.info("🔧 VLM: DISABLED (structure test only)")
    
    # Parse without VLM
    parser = HybridPDFParser(
        config={"pdfplumber_mode": 0},  # Use SmolDocling
        enable_vlm=False  # No VLM for this test
    )
    
    try:
        start_time = datetime.now()
        document = await parser.parse(test_file)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Parsing completed in {parse_time:.2f}s")
        logger.info(f"📊 Document Statistics:")
        logger.info(f"   - Total segments: {len(document.segments)}")
        logger.info(f"   - Visual elements: {len(document.visual_elements)}")
        logger.info(f"   - Pages: {document.metadata.page_count}")
        
        # Analyze segment types
        segment_stats = {}
        for seg in document.segments:
            key = f"{seg.segment_type.value}/{seg.segment_subtype or 'none'}"
            segment_stats[key] = segment_stats.get(key, 0) + 1
        
        logger.info("\n📊 Segment Distribution:")
        for key, count in sorted(segment_stats.items()):
            logger.info(f"   - {key}: {count}")
        
        # Show visual segments
        visual_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.VISUAL]
        logger.info(f"\n🎨 Visual Segments: {len(visual_segments)}")
        
        for i, seg in enumerate(visual_segments[:10]):  # First 10
            logger.info(f"\n  Visual Segment {i+1}:")
            logger.info(f"    - Page: {seg.page_number}")
            logger.info(f"    - Subtype: {seg.segment_subtype}")
            logger.info(f"    - Content: {seg.content}")
            logger.info(f"    - Visual refs: {seg.visual_references}")
        
        # Save results
        output_dir = Path("tests/debugging/vlm_integration")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON output
        segments_data = []
        for seg in document.segments:
            seg_data = {
                "index": seg.segment_index,
                "type": seg.segment_type.value,
                "subtype": seg.segment_subtype,
                "page": seg.page_number,
                "content": seg.content[:200] + "..." if len(seg.content) > 200 else seg.content,
                "visual_refs": seg.visual_references
            }
            segments_data.append(seg_data)
        
        output = {
            "file": str(test_file),
            "timestamp": datetime.now().isoformat(),
            "parse_time_seconds": parse_time,
            "statistics": {
                "total_segments": len(document.segments),
                "visual_elements": len(document.visual_elements),
                "visual_segments": len(visual_segments),
                "pages": document.metadata.page_count,
                "segment_distribution": segment_stats
            },
            "segments": segments_data
        }
        
        json_file = output_dir / f"bmw_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to: {json_file}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
    finally:
        if hasattr(parser, 'cleanup'):
            parser.cleanup()


if __name__ == "__main__":
    asyncio.run(test_bmw_structure())