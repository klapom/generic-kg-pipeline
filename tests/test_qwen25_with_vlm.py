#!/usr/bin/env python3
"""
Test Qwen2.5-VL Integration with Visual Elements

This test verifies VLM processing on a document with actual visual elements.
"""

import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25

# Setup logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'qwen25_vlm_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def test_vlm_integration():
    """Test VLM integration with visual elements"""
    
    # Test with BMW document which has images
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        logger.info("=" * 80)
        logger.info("🧠 TESTING QWEN2.5-VL WITH VISUAL ELEMENTS")
        logger.info("=" * 80)
        
        # Test with VLM enabled
        logger.info("\n📄 Parsing BMW document with Qwen2.5-VL...")
        parser = HybridPDFParserQwen25(
            config={
                "max_pages": 3,  # Limit for testing
                "pdfplumber_mode": 1,
                "enable_page_context": False,  # Disable for now
                "vlm": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "batch_size": 2,
                    "enable_structured_parsing": True
                },
                "image_extraction": {
                    "min_size": 100,
                    "extract_embedded": True,
                    "render_fallback": True
                }
            },
            enable_vlm=True
        )
        
        document = await parser.parse(test_file)
        
        logger.info(f"\n✅ Parsing completed")
        logger.info(f"   Total segments: {len(document.segments)}")
        logger.info(f"   Visual elements: {len(document.visual_elements)}")
        
        # Analyze visual elements
        vlm_analyzed = 0
        structured_data_count = 0
        
        logger.info("\n📊 VISUAL ELEMENT ANALYSIS:")
        logger.info("-" * 60)
        
        for i, ve in enumerate(document.visual_elements[:10]):  # First 10
            logger.info(f"\n🖼️ Visual Element {i+1}:")
            logger.info(f"   Type: {ve.element_type.value}")
            logger.info(f"   Page: {ve.page_or_slide}")
            logger.info(f"   Has raw data: {bool(ve.raw_data)}")
            
            if ve.vlm_description:
                vlm_analyzed += 1
                logger.info(f"   ✅ VLM Description: {ve.vlm_description[:100]}...")
                
                if hasattr(ve, 'analysis_metadata') and ve.analysis_metadata.get('structured_data'):
                    structured_data_count += 1
                    struct_type = ve.analysis_metadata['structured_data'].get('type')
                    logger.info(f"   📋 Structured data type: {struct_type}")
            else:
                logger.info(f"   ❌ No VLM description")
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"   VLM analyzed: {vlm_analyzed}/{len(document.visual_elements)}")
        logger.info(f"   With structured data: {structured_data_count}")
        
        # Check visual segments
        visual_segments = [s for s in document.segments if s.segment_type.value == 'visual']
        vlm_updated_segments = [s for s in visual_segments if s.metadata.get('vlm_analyzed')]
        
        logger.info(f"\n📝 Visual Segments:")
        logger.info(f"   Total visual segments: {len(visual_segments)}")
        logger.info(f"   Updated with VLM: {len(vlm_updated_segments)}")
        
        # Save results
        results = {
            "file": str(test_file),
            "total_segments": len(document.segments),
            "visual_elements": len(document.visual_elements),
            "vlm_analyzed": vlm_analyzed,
            "structured_data_count": structured_data_count,
            "visual_segments": len(visual_segments),
            "vlm_updated_segments": len(vlm_updated_segments),
            "sample_descriptions": [
                {
                    "type": ve.element_type.value,
                    "page": ve.page_or_slide,
                    "description": ve.vlm_description[:200] if ve.vlm_description else None
                }
                for ve in document.visual_elements[:5]
            ]
        }
        
        output_file = log_dir / f"qwen25_vlm_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        parser.cleanup()
        
        # Success if we analyzed at least some visual elements
        return vlm_analyzed > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_vlm_integration())
    print(f"\nTest {'passed' if success else 'failed'}!")