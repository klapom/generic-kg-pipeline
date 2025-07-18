#!/usr/bin/env python3
"""
Simple test for Qwen2.5-VL integration
"""

import asyncio
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple():
    """Simple test to verify basic functionality"""
    
    # Test file
    test_file = Path("data/input/test_simple.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        # Test without VLM first
        logger.info("Testing parser without VLM...")
        parser = HybridPDFParserQwen25(
            config={
                "max_pages": 5,
                "pdfplumber_mode": 1
            },
            enable_vlm=False
        )
        
        document = await parser.parse(test_file)
        logger.info(f"✅ Parsed successfully: {len(document.segments)} segments")
        
        parser.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple())
    print(f"\nTest {'passed' if success else 'failed'}!")