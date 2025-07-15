#!/usr/bin/env python3
"""
Quick test to check bbox extraction
"""

import re
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_bbox_parsing():
    """Test bbox parsing from SmolDocling output"""
    
    # Sample SmolDocling output from the BMW document
    sample_output = """<doctag><page_header><loc_31><loc_11><loc_37><loc_16>3</page_header>
<page_header><loc_46><loc_11><loc_290><loc_18>AUDI AG | I/EG-GE2 | Wettbewerbsanalyse Preview - BMW 3er G20 | Burkhard Fuchs, Marvin Gregor | März 2019</page_header>
<section_header_level_1><loc_35><loc_34><loc_168><loc_52>Ansichten (Exterieur) I</section_header_level_1>
<text><loc_35><loc_62><loc_189><loc_72>Preview - BMW 3er G20 (ME: 09.März.2019)</text>
<text><loc_336><loc_62><loc_424><loc_71>Quellen: press.bmwgroup.com; NetCarShow</text>
<section_header_level_1><loc_25><loc_96><loc_90><loc_108>Vorgänger (F30)</section_header_level_1>
<picture><loc_24><loc_116><loc_195><loc_322></picture>
<picture><loc_24><loc_328><loc_224><loc_483></picture>
<picture><loc_290><loc_116><loc_493><loc_322></picture>
<picture><loc_290><loc_334><loc_493><loc_483></picture>
<picture><loc_430><loc_25><loc_491><loc_60></picture>
</doctag>"""
    
    logger.info("Testing bbox parsing from SmolDocling output...")
    
    # Create client and test parse_model_output
    client = VLLMSmolDoclingClient(auto_load=False)
    
    # Test the parse_model_output method
    parsed = client.parse_model_output(sample_output)
    
    logger.info(f"\nParsed data:")
    logger.info(f"  - Text blocks: {len(parsed.get('text_blocks', []))}")
    logger.info(f"  - Tables: {len(parsed.get('tables', []))}")
    logger.info(f"  - Images: {len(parsed.get('images', []))}")
    
    # Check images and their bbox
    images = parsed.get('images', [])
    logger.info(f"\n🖼️ Found {len(images)} images:")
    
    for i, img in enumerate(images, 1):
        logger.info(f"\n  Image {i}:")
        logger.info(f"    - Content: {img.get('content', '')[:50]}...")
        logger.info(f"    - Caption: {img.get('caption', 'No caption')}")
        logger.info(f"    - BBox: {img.get('bbox', 'NO BBOX FOUND!')}")
        
        # Check if bbox has 4 values
        bbox = img.get('bbox')
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            logger.info(f"    ✅ Valid bbox with 4 coordinates")
        else:
            logger.error(f"    ❌ Invalid bbox: {bbox}")
    
    # Test creating visual elements
    logger.info("\n\n🔧 Testing Visual Element creation...")
    
    # Create a mock SmolDoclingPage
    from core.clients.vllm_smoldocling_local import SmolDoclingPage
    
    page = SmolDoclingPage(
        page_number=3,
        text=parsed.get("text", ""),
        tables=parsed.get("tables", []),
        images=parsed.get("images", []),
        formulas=parsed.get("formulas", []),
        layout_info=parsed.get("layout_info", {}),
        confidence_score=parsed.get("confidence_score", 0.0)
    )
    
    # Create a mock SmolDoclingResult
    from core.clients.vllm_smoldocling_local import SmolDoclingResult
    
    result = SmolDoclingResult(
        pages=[page],
        metadata={},
        processing_time_seconds=1.0,
        model_version="test",
        total_pages=1,
        success=True
    )
    
    # Convert to document
    logger.info("\nConverting to Document...")
    document = client.convert_to_document(result, Path("test.pdf"))
    
    logger.info(f"\n📊 Document created:")
    logger.info(f"  - Segments: {len(document.segments)}")
    logger.info(f"  - Visual Elements: {len(document.visual_elements)}")
    
    # Check visual elements
    logger.info(f"\n🎯 Visual Elements bbox check:")
    for i, ve in enumerate(document.visual_elements, 1):
        logger.info(f"\n  Visual Element {i}:")
        logger.info(f"    - Type: {ve.element_type.value}")
        logger.info(f"    - Page: {ve.page_or_slide}")
        logger.info(f"    - Has BBox: {ve.bounding_box is not None}")
        if ve.bounding_box:
            logger.info(f"    - BBox: {ve.bounding_box}")
        
        # Check raw_bbox in metadata
        if ve.analysis_metadata and 'raw_bbox' in ve.analysis_metadata:
            logger.info(f"    - Raw BBox: {ve.analysis_metadata['raw_bbox']}")


if __name__ == "__main__":
    test_bbox_parsing()