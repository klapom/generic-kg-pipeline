#!/usr/bin/env python3
"""
Debug test for BMW PDF page 2 - why is content missing?
"""

import logging
from pathlib import Path
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_page2():
    """Test SmolDocling on page 2 specifically"""
    try:
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        
        logger.info("Creating SmolDocling client...")
        client = VLLMSmolDoclingClient(
            max_pages=1,  # Only process page 2
            gpu_memory_utilization=0.25  # More memory for complex page
        )
        
        # Load the model first
        logger.info("Loading SmolDocling model...")
        from core.vllm.model_manager import VLLMModelManager
        model_manager = VLLMModelManager()
        model_manager.load_model(client.model_id)
        
        # Override to start at page 2
        logger.info("Processing ONLY page 2...")
        
        # Convert specific page to image
        import pdf2image
        from PIL import Image
        
        pages = pdf2image.convert_from_path(
            pdf_path, 
            first_page=2, 
            last_page=2,
            dpi=300  # High quality
        )
        
        if not pages:
            logger.error("Failed to convert page 2")
            return
            
        page_image = pages[0]
        logger.info(f"Page 2 image size: {page_image.size}")
        
        # Process single page directly
        logger.info("Processing page 2 with SmolDocling...")
        page_result = client.process_pdf_page(page_image, 2)
        
        logger.info("="*80)
        logger.info("PAGE 2 DETAILED ANALYSIS:")
        logger.info("="*80)
        logger.info(f"Text extracted: {len(page_result.text)} chars")
        logger.info(f"Tables found: {len(page_result.tables)}")
        logger.info(f"Images found: {len(page_result.images)}")
        logger.info(f"Layout info: {page_result.layout_info}")
        
        # Check raw vLLM response
        if 'vllm_response' in page_result.layout_info:
            raw = page_result.layout_info['vllm_response']
            logger.info(f"\nRAW vLLM RESPONSE LENGTH: {len(raw)} chars")
            logger.info("RAW RESPONSE:")
            logger.info("-"*80)
            logger.info(raw)
            logger.info("-"*80)
            
            # Check for specific elements we expect
            logger.info("\nCHECKING FOR EXPECTED ELEMENTS:")
            logger.info(f"- Contains 'Motorisierungen': {'Motorisierungen' in raw}")
            logger.info(f"- Contains 'table': {'<table>' in raw}")
            logger.info(f"- Contains 'Highlights': {'Highlights' in raw}")
            logger.info(f"- Contains '320d': {'320d' in raw}")
            logger.info(f"- Contains 'Weltpremiere': {'Weltpremiere' in raw}")
            
        return page_result
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("Starting page 2 debug test...")
    result = test_page2()
    
    if result:
        logger.info(f"\nFINAL RESULT: Extracted {len(result.text)} chars from page 2")
        if result.text:
            logger.info("EXTRACTED TEXT:")
            logger.info(result.text)
    else:
        logger.error("Test failed!")