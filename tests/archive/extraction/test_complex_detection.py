#!/usr/bin/env python3
"""
Test complex layout detection with BMW PDF
"""

import logging
from pathlib import Path
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_complex_detection():
    """Test complex layout detection on all pages"""
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("🚀 Testing complex layout detection...")
    
    # Create client (this will register the model)
    client = VLLMSmolDoclingClient(
        max_pages=5,  # Test all 5 pages
        gpu_memory_utilization=0.2
    )
    
    # Parse PDF
    result = client.parse_pdf(pdf_path)
    
    if result.success:
        logger.info(f"✅ PDF parsing successful! Processed {len(result.pages)} pages")
        
        # Analyze each page
        complex_pages = []
        for page in result.pages:
            detection = page.layout_info.get('complex_layout_detection', {})
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 Page {page.page_number} Analysis:")
            logger.info(f"{'='*80}")
            
            if detection.get('is_complex_layout', False):
                complex_pages.append(page.page_number)
                logger.warning(f"🚨 COMPLEX LAYOUT DETECTED!")
                logger.info(f"Detection details:")
                for key, value in detection['detection_details'].items():
                    logger.info(f"  - {key}: {value}")
                logger.info(f"  - Confidence: {detection['confidence']}")
                logger.info(f"  ➡️  NEEDS FALLBACK PARSER!")
            else:
                logger.info(f"✅ Normal layout - SmolDocling handled well")
                logger.info(f"  - Text: {len(page.text)} chars")
                logger.info(f"  - Tables: {len(page.tables)}")
                logger.info(f"  - Images: {len(page.images)}")
                logger.info(f"  - Formulas: {len(page.formulas)}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 SUMMARY:")
        logger.info(f"{'='*80}")
        logger.info(f"Total pages: {len(result.pages)}")
        logger.info(f"Complex pages detected: {len(complex_pages)} - {complex_pages}")
        logger.info(f"Normal pages: {len(result.pages) - len(complex_pages)}")
        
        # Test our hypothesis about page 2
        if 2 in complex_pages:
            logger.info(f"\n✅ SUCCESS! Page 2 correctly identified as complex layout!")
        else:
            logger.warning(f"\n⚠️  Page 2 NOT detected as complex. May need to adjust thresholds.")
            
        return complex_pages
    else:
        logger.error(f"❌ PDF parsing failed: {result.error_message}")
        return []

if __name__ == "__main__":
    # Load model first
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model("smoldocling")
    
    # Run test
    complex_pages = test_complex_detection()