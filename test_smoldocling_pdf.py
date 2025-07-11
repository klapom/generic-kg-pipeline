#!/usr/bin/env python3
"""
Test SmolDocling PDF processing directly
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_smoldocling_pdf():
    """Test SmolDocling PDF processing directly"""
    try:
        logger.info("🧪 Testing SmolDocling PDF processing...")
        
        # Find PDF file - use BMW PDF
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            return False
            
        # Import and create SmolDocling client
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        
        logger.info("📦 Creating SmolDocling client...")
        client = VLLMSmolDoclingClient(max_pages=2)  # Process 2 pages for focused testing
        
        logger.info("🔄 Processing PDF with SmolDocling...")
        result = client.parse_pdf(pdf_path)
        
        if result.success:
            logger.info(f"✅ PDF processing successful!")
            logger.info(f"📊 Pages processed: {len(result.pages)}")
            logger.info(f"⏱️ Processing time: {result.processing_time_seconds:.1f}s")
            
            # Show content from all pages - raw V2T output
            for i, page in enumerate(result.pages, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"📄 PAGE {i} - RAW V2T OUTPUT:")
                logger.info(f"{'='*80}")
                logger.info(f"📝 Text content ({len(page.text)} chars):")
                logger.info(f"--- START RAW TEXT ---")
                logger.info(page.text)
                logger.info(f"--- END RAW TEXT ---")
                
                if page.tables:
                    logger.info(f"\n📊 Tables ({len(page.tables)}):")
                    for j, table in enumerate(page.tables, 1):
                        logger.info(f"--- TABLE {j} START ---")
                        logger.info(f"{table}")
                        logger.info(f"--- TABLE {j} END ---")
                
                if page.images:
                    logger.info(f"\n🖼️ Images ({len(page.images)}):")
                    for j, img in enumerate(page.images, 1):
                        logger.info(f"--- IMAGE {j} START ---")
                        logger.info(f"{img}")
                        logger.info(f"--- IMAGE {j} END ---")
                        
                if page.formulas:
                    logger.info(f"\n🧮 Formulas ({len(page.formulas)}):")
                    for j, formula in enumerate(page.formulas, 1):
                        logger.info(f"--- FORMULA {j} START ---")
                        logger.info(f"{formula}")
                        logger.info(f"--- FORMULA {j} END ---")
                
                # Show layout info with raw content
                logger.info(f"\n🔍 Layout Info:")
                if page.layout_info and 'raw_content' in page.layout_info:
                    logger.info(f"Raw content length: {len(page.layout_info['raw_content'])} chars")
                    logger.info(f"Confidence score: {page.confidence_score}")
                else:
                    logger.info(f"Layout info: {page.layout_info}")
                    logger.info(f"Confidence score: {page.confidence_score}")
                
            # Test conversion to Document
            logger.info("🔄 Converting to Document format...")
            document = client.convert_to_document(result, pdf_path)
            logger.info(f"✅ Document created: {len(document.segments)} segments")
            
            return True
        else:
            logger.error(f"❌ PDF processing failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting SmolDocling PDF test...")
    success = test_smoldocling_pdf()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")