#!/usr/bin/env python3
"""
Test PDF processing with single page
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_page_pdf():
    """Test PDF processing with just one page"""
    try:
        logger.info("🧪 Testing single page PDF processing...")
        
        # Find PDF file
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            return False
            
        # Import and create PDF parser
        from plugins.parsers.pdf_parser import PDFParser
        
        logger.info("📦 Creating PDF parser...")
        parser = PDFParser({"enable_vlm": True})
        
        logger.info("🔄 Processing single page...")
        # Limit to 1 page for testing
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        client = VLLMSmolDoclingClient(max_pages=1)
        
        # Test convert PDF to images
        from pdf2image import convert_from_path
        page_images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
        logger.info(f"✅ Converted 1 page to image")
        
        # Test processing single page  
        result = client.process_pdf_page(page_images[0], 1)
        logger.info(f"✅ Page processed: {len(result.text)} chars")
        logger.info(f"📝 Text preview: {result.text[:200]}...")
        
        return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting single page PDF test...")
    success = test_single_page_pdf()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")