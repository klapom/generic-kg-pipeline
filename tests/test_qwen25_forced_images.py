#!/usr/bin/env python3
"""
Test Qwen2.5-VL with Forced Image Extraction

This test forces image extraction to verify VLM processing works correctly.
"""

import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime
import fitz  # PyMuPDF
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25
from core.parsers.interfaces import VisualElement, VisualElementType, DocumentType

# Setup logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'qwen25_forced_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def force_extract_images(pdf_path: Path, max_pages: int = 3):
    """Force extract images from PDF as visual elements"""
    visual_elements = []
    
    doc = fitz.open(str(pdf_path))
    
    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        
        # Extract page as image
        mat = fitz.Matrix(2, 2)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Create visual element
        content_hash = hashlib.sha256(img_data).hexdigest()[:16]
        
        ve = VisualElement(
            element_type=VisualElementType.IMAGE,
            source_format=DocumentType.PDF,
            content_hash=content_hash,
            page_or_slide=page_num + 1,
            raw_data=img_data,
            bounding_box=[0, 0, pix.width, pix.height],
            analysis_metadata={
                "source": "forced_page_extraction",
                "original_size": [pix.width, pix.height]
            }
        )
        visual_elements.append(ve)
        logger.info(f"✅ Extracted page {page_num + 1} as image ({pix.width}x{pix.height})")
    
    doc.close()
    return visual_elements


async def test_forced_vlm():
    """Test VLM with forced image extraction"""
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        logger.info("=" * 80)
        logger.info("🧠 TESTING QWEN2.5-VL WITH FORCED IMAGES")
        logger.info("=" * 80)
        
        # First, parse without VLM to get base document
        logger.info("\n📄 Step 1: Parse document structure...")
        parser = HybridPDFParserQwen25(
            config={
                "max_pages": 3,
                "pdfplumber_mode": 1
            },
            enable_vlm=False
        )
        
        document = await parser.parse(test_file)
        logger.info(f"✅ Base parsing completed: {len(document.segments)} segments")
        
        # Force extract images
        logger.info("\n🖼️ Step 2: Force extract images...")
        forced_visuals = force_extract_images(test_file, max_pages=3)
        logger.info(f"✅ Extracted {len(forced_visuals)} images")
        
        # Add forced visuals to document
        document.visual_elements = forced_visuals
        
        # Now process with VLM
        logger.info("\n🧠 Step 3: Process with Qwen2.5-VL...")
        
        # Initialize VLM processor
        from core.vlm.qwen25_processor import Qwen25VLMProcessor
        
        vlm_processor = Qwen25VLMProcessor({
            'temperature': 0.2,
            'max_new_tokens': 256,
            'batch_size': 2,
            'enable_page_context': False,
            'enable_structured_parsing': True
        })
        
        # Process visual elements
        analysis_results = await vlm_processor.process_visual_elements(
            document.visual_elements
        )
        
        # Update visual elements with results
        success_count = 0
        for ve, result in zip(document.visual_elements, analysis_results):
            if result.success:
                ve.vlm_description = result.description
                ve.confidence_score = result.confidence
                ve.ocr_text = result.ocr_text
                success_count += 1
                
                logger.info(f"\n✅ Page {ve.page_or_slide} analyzed:")
                logger.info(f"   Description: {result.description[:150]}...")
                if result.structured_data:
                    logger.info(f"   Structured data type: {result.structured_data.get('type')}")
            else:
                logger.error(f"❌ Page {ve.page_or_slide} failed: {result.error_message}")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Successfully analyzed: {success_count}/{len(document.visual_elements)}")
        
        # Cleanup
        vlm_processor.cleanup()
        parser.cleanup()
        
        # Save results
        results = {
            "test": "forced_image_extraction",
            "file": str(test_file),
            "pages_processed": len(forced_visuals),
            "vlm_success": success_count,
            "sample_descriptions": [
                {
                    "page": ve.page_or_slide,
                    "description": ve.vlm_description[:200] if ve.vlm_description else None
                }
                for ve in document.visual_elements
            ]
        }
        
        output_file = log_dir / f"qwen25_forced_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_forced_vlm())
    print(f"\nTest {'passed' if success else 'failed'}!")