#!/usr/bin/env python3
"""
Test SmolDocling parsing to debug the coordinate prefix issue
"""

import asyncio
import logging
from pathlib import Path
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_smoldocling_parsing():
    """Test SmolDocling parsing with actual PDF"""
    
    # Initialize SmolDocling client
    client = VLLMSmolDoclingClient(max_pages=1)
    
    # Test with simple PDF
    pdf_path = Path("data/input/test_simple.pdf")
    if not pdf_path.exists():
        logger.error(f"Test file not found: {pdf_path}")
        return
        
    logger.info(f"\n{'='*60}")
    logger.info("Testing SmolDocling Parsing")
    logger.info(f"{'='*60}\n")
    
    try:
        # Parse PDF
        result = client.parse_pdf(pdf_path)
        
        logger.info(f"\nParsing Result:")
        logger.info(f"Success: {result.success}")
        logger.info(f"Pages: {len(result.pages)}")
        
        if result.pages:
            page = result.pages[0]
            logger.info(f"\nPage 1 Content:")
            logger.info(f"Raw text length: {len(page.text)}")
            logger.info(f"Text preview: {page.text[:200]}...")
            
            # Check if text contains coordinate prefixes
            import re
            coord_pattern = r'^\d+>\d+>\d+>\d+>'
            if re.search(coord_pattern, page.text, re.MULTILINE):
                logger.warning("\n⚠️ FOUND COORDINATE PREFIXES IN TEXT!")
                logger.warning("This means _clean_coordinate_prefix is not working properly")
                
                # Show examples
                lines = page.text.split('\n')
                for line in lines[:5]:
                    if re.match(coord_pattern, line):
                        logger.warning(f"Example: {line[:50]}...")
            else:
                logger.info("\n✅ No coordinate prefixes found - parsing is correct!")
                
    except Exception as e:
        logger.error(f"Error during parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_smoldocling_parsing())