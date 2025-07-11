#!/usr/bin/env python3
"""
Focused test for page 2 complex detection
"""

import logging
import json
from pathlib import Path
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

# Set up detailed logging  
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Only show INFO and above for vLLM/urllib3
logging.getLogger('vllm').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

def test_page2():
    """Test page 2 specifically"""
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("Creating SmolDocling client...")
    client = VLLMSmolDoclingClient(max_pages=1, gpu_memory_utilization=0.2)
    
    logger.info("Loading model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(client.model_id)
    
    # Convert page 2 to image
    import pdf2image
    pages = pdf2image.convert_from_path(pdf_path, first_page=2, last_page=2, dpi=300)
    if not pages:
        logger.error("Failed to convert page 2")
        return
        
    page_image = pages[0]
    logger.info(f"Processing page 2 (size: {page_image.size})...")
    
    # Process the page
    page_result = client.process_pdf_page(page_image, 2)
    
    # Check detection
    detection = page_result.layout_info.get('complex_layout_detection', {})
    
    logger.info("\n" + "="*80)
    logger.info("PAGE 2 COMPLEX LAYOUT DETECTION RESULTS:")
    logger.info("="*80)
    
    logger.info(f"Is Complex Layout: {detection.get('is_complex_layout', False)}")
    logger.info(f"Confidence: {detection.get('confidence', 0)}")
    
    logger.info("\nDetection Details:")
    details = detection.get('detection_details', {})
    for key, value in details.items():
        logger.info(f"  {key}: {value}")
    
    logger.info(f"\nExtracted Content:")
    logger.info(f"  Text length: {len(page_result.text)} chars")
    logger.info(f"  Tables: {len(page_result.tables)}")
    logger.info(f"  Images: {len(page_result.images)}")
    
    if detection.get('is_complex_layout', False):
        logger.info("\n✅ SUCCESS! Page 2 correctly identified as complex layout!")
        logger.info("➡️  This page needs fallback parser (PyPDF2/pdfplumber)")
    else:
        logger.warning("\n⚠️  Page 2 NOT detected as complex. Checking why...")
        logger.info("\nRaw content preview:")
        raw = page_result.layout_info.get('raw_content', '')
        logger.info(raw[:200] + "..." if len(raw) > 200 else raw)
        
    # Save detection results
    with open('data/output/page2_detection.json', 'w') as f:
        json.dump({
            'detection': detection,
            'page_info': {
                'text_length': len(page_result.text),
                'tables': len(page_result.tables),
                'images': len(page_result.images),
                'raw_content_length': len(page_result.layout_info.get('raw_content', ''))
            }
        }, f, indent=2)
    logger.info(f"\nDetection results saved to: data/output/page2_detection.json")

if __name__ == "__main__":
    test_page2()