#!/usr/bin/env python3
"""
Minimal test to debug string index error
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser

# Set up logging to console with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Set specific loggers to DEBUG
logging.getLogger('core.parsers.implementations.pdf.hybrid_pdf_parser').setLevel(logging.DEBUG)
logging.getLogger('core.clients.vllm_smoldocling_local').setLevel(logging.INFO)


async def test_minimal():
    """Minimal test to find string index error"""
    print("=" * 80)
    print("DEBUG: String Index Error Test")
    print("=" * 80)
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    try:
        parser = HybridPDFParser(
            config={"pdfplumber_mode": 0},
            enable_vlm=False
        )
        
        print(f"\n📖 Parsing: {test_file}")
        document = await parser.parse(test_file)
        
        print(f"\n✅ SUCCESS! Document parsed:")
        print(f"   - Segments: {len(document.segments)}")
        print(f"   - Visual elements: {len(document.visual_elements)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Re-raise to see full stack trace
        raise


if __name__ == "__main__":
    asyncio.run(test_minimal())