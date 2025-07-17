#\!/usr/bin/env python3
"""Test HybridPDFParser with unified client"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

from core.parsers.implementations.pdf.hybrid_pdf_parser import HybridPDFParser
import asyncio

async def test_parser():
    """Test PDF parsing with HybridPDFParser"""
    print("\n=== Testing HybridPDFParser with Unified Client ===\n")
    
    # Create parser
    config = {
        'environment': 'production',
        'max_pages': 5,
        'use_docling_final': True  # This should not be needed anymore
    }
    
    parser = HybridPDFParser(config=config, enable_vlm=False)
    print(f"✅ Parser created")
    print(f"   Client type: {type(parser.smoldocling_client).__name__}")
    
    # Test PDF
    test_pdf = Path("data/input/test_simple.pdf")
    if not test_pdf.exists():
        test_pdf = Path("data/input/Preview_BMW_1er_Sedan_CN.pdf")
    
    if test_pdf.exists():
        print(f"\n📄 Testing with: {test_pdf.name}")
        try:
            document = await parser.parse(test_pdf)
            print(f"✅ Document parsed successfully")
            print(f"   Total segments: {document.total_segments}")
            print(f"   Total visual elements: {document.total_visual_elements}")
            print(f"   Pages: {len(document.pages)}")
            
            if document.pages:
                page = document.pages[0]
                print(f"\n   Page 1 segments: {len(page.segments)}")
                for i, seg in enumerate(page.segments[:3]):
                    print(f"   Segment {i}: {seg.type.value} - {seg.text[:50] if seg.text else 'No text'}...")
                    
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_parser())
