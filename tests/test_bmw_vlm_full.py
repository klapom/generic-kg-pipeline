#!/usr/bin/env python3
"""
Test VLM Integration with BMW 3er Document

Full test with detailed logging and JSON output
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

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/debugging/vlm_integration/bmw_vlm_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_bmw_document_with_vlm():
    """Test BMW document with full VLM integration"""
    logger.info("=" * 80)
    logger.info("BMW 3er Document VLM Integration Test")
    logger.info("=" * 80)
    
    # Test file
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"📄 Processing: {test_file.name}")
    logger.info(f"🔧 VLM Integration: ENABLED")
    
    # Create output directory
    output_dir = Path("tests/debugging/vlm_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize parser with VLM enabled
    parser = HybridPDFParser(
        config={
            "pdfplumber_mode": 0,  # Use SmolDocling for visual extraction
            "log_level": "DEBUG"
        },
        enable_vlm=True
    )
    
    try:
        # Parse document
        logger.info("\n📖 Starting document parsing...")
        start_time = datetime.now()
        
        document = await parser.parse(test_file)
        
        parse_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Parsing completed in {parse_time:.2f} seconds")
        
        # Document statistics
        logger.info("\n📊 Document Statistics:")
        logger.info(f"   - Total segments: {len(document.segments)}")
        logger.info(f"   - Total visual elements: {len(document.visual_elements)}")
        logger.info(f"   - Pages: {document.metadata.page_count}")
        
        # Analyze segments by type
        segment_stats = {}
        for seg in document.segments:
            seg_type = seg.segment_type.value
            seg_subtype = seg.segment_subtype or "none"
            key = f"{seg_type}/{seg_subtype}"
            segment_stats[key] = segment_stats.get(key, 0) + 1
        
        logger.info("\n📊 Segment Distribution:")
        for key, count in sorted(segment_stats.items()):
            logger.info(f"   - {key}: {count}")
        
        # Visual segments analysis
        visual_segments = [seg for seg in document.segments if seg.segment_type == SegmentType.VISUAL]
        logger.info(f"\n🎨 Visual Segments: {len(visual_segments)}")
        
        vlm_analyzed = 0
        placeholders = 0
        
        logger.info("\n🔍 Visual Segment Details:")
        for i, seg in enumerate(visual_segments):
            is_placeholder = seg.content.startswith("[") and seg.content.endswith("]")
            is_analyzed = seg.metadata and seg.metadata.get("vlm_analyzed", False)
            
            if is_placeholder:
                placeholders += 1
            if is_analyzed:
                vlm_analyzed += 1
            
            logger.info(f"\n  Visual Segment {i+1} (Page {seg.page_number}):")
            logger.info(f"    - Subtype: {seg.segment_subtype}")
            logger.info(f"    - VLM Analyzed: {is_analyzed}")
            logger.info(f"    - Is Placeholder: {is_placeholder}")
            
            if is_analyzed:
                logger.info(f"    - Confidence: {seg.metadata.get('confidence', 0):.2f}")
                logger.info(f"    - Has Extracted Data: {seg.metadata.get('has_extracted_data', False)}")
            
            # Log content (truncated)
            content_preview = seg.content[:200] + "..." if len(seg.content) > 200 else seg.content
            logger.info(f"    - Content: {content_preview}")
            
            # Visual reference
            if seg.visual_references:
                logger.info(f"    - Visual Reference: {seg.visual_references[0]}")
        
        # Visual elements analysis
        logger.info(f"\n🖼️ Visual Elements: {len(document.visual_elements)}")
        for i, ve in enumerate(document.visual_elements[:10]):  # First 10
            logger.info(f"\n  Visual Element {i+1} (Page {ve.page_or_slide}):")
            logger.info(f"    - Type: {ve.element_type.value}")
            logger.info(f"    - Has VLM Description: {bool(ve.vlm_description)}")
            logger.info(f"    - Confidence: {ve.confidence:.2f}")
            logger.info(f"    - Content Hash: {ve.content_hash}")
            
            if ve.vlm_description:
                desc_preview = ve.vlm_description[:150] + "..." if len(ve.vlm_description) > 150 else ve.vlm_description
                logger.info(f"    - Description: {desc_preview}")
            
            if ve.extracted_data:
                logger.info(f"    - Extracted Data Keys: {list(ve.extracted_data.keys())}")
        
        # Summary statistics
        logger.info("\n📊 VLM Analysis Summary:")
        logger.info(f"   - Total visual segments: {len(visual_segments)}")
        logger.info(f"   - Placeholders remaining: {placeholders}")
        logger.info(f"   - VLM analyzed segments: {vlm_analyzed}")
        if len(visual_segments) > 0:
            logger.info(f"   - Analysis success rate: {vlm_analyzed/len(visual_segments)*100:.1f}%")
        
        # Create comprehensive JSON output
        json_output = {
            "file": str(test_file),
            "parse_time_seconds": parse_time,
            "timestamp": datetime.now().isoformat(),
            "document_metadata": {
                "title": document.metadata.title,
                "page_count": document.metadata.page_count,
                "file_size": document.metadata.file_size,
                "document_type": document.metadata.document_type.value
            },
            "statistics": {
                "total_segments": len(document.segments),
                "total_visual_elements": len(document.visual_elements),
                "visual_segments": len(visual_segments),
                "vlm_analyzed": vlm_analyzed,
                "placeholders": placeholders,
                "segment_distribution": segment_stats
            },
            "segments": []
        }
        
        # Add all segments to JSON
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
            json_output["segments"].append(seg_data)
        
        # Add visual elements
        json_output["visual_elements"] = []
        for ve in document.visual_elements:
            ve_data = {
                "element_type": ve.element_type.value,
                "page": ve.page_or_slide,
                "content_hash": ve.content_hash,
                "confidence": ve.confidence,
                "has_vlm_description": bool(ve.vlm_description),
                "vlm_description": ve.vlm_description,
                "extracted_data": ve.extracted_data,
                "bounding_box": ve.bounding_box,
                "analysis_metadata": ve.analysis_metadata
            }
            json_output["visual_elements"].append(ve_data)
        
        # Save JSON output
        json_file = output_dir / f"bmw_vlm_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to:")
        logger.info(f"   - JSON: {json_file}")
        logger.info(f"   - Log: tests/debugging/vlm_integration/bmw_vlm_test.log")
        
        # Create a simplified summary for quick viewing
        summary = {
            "summary": {
                "file": test_file.name,
                "pages": document.metadata.page_count,
                "visual_elements_found": len(document.visual_elements),
                "visual_segments_created": len(visual_segments),
                "vlm_analysis_success_rate": f"{vlm_analyzed/len(visual_segments)*100:.1f}%" if visual_segments else "N/A"
            },
            "sample_visual_descriptions": []
        }
        
        # Add sample descriptions
        for seg in visual_segments[:5]:  # First 5 visual segments
            if seg.metadata and seg.metadata.get("vlm_analyzed"):
                summary["sample_visual_descriptions"].append({
                    "page": seg.page_number,
                    "type": seg.segment_subtype,
                    "description": seg.content[:200] + "..." if len(seg.content) > 200 else seg.content
                })
        
        summary_file = output_dir / "bmw_vlm_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   - Summary: {summary_file}")
        
        logger.info("\n✅ BMW VLM Integration Test COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if hasattr(parser, 'cleanup'):
            parser.cleanup()
        logger.info("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_bmw_document_with_vlm())