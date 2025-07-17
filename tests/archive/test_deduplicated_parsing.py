#!/usr/bin/env python3
"""
Test to verify image deduplication works
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_deduplication():
    """Test that image deduplication reduces visual elements count"""
    print("=" * 80)
    print("Testing Image Deduplication")
    print("=" * 80)
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    parser = HybridPDFParser(
        config={"pdfplumber_mode": 0},
        enable_vlm=False
    )
    
    print(f"\n📖 Parsing: {test_file}")
    document = await parser.parse(test_file)
    
    print(f"\n✅ Document parsed successfully!")
    print(f"   - Total segments: {len(document.segments)}")
    print(f"   - Visual elements: {len(document.visual_elements)}")
    print(f"   - Visual segments: {len([s for s in document.segments if s.segment_type.value == 'visual'])}")
    
    # Check for improvement
    if len(document.visual_elements) < 899:
        print(f"\n🎉 DEDUPLICATION SUCCESSFUL!")
        print(f"   Visual elements reduced from 899 to {len(document.visual_elements)}")
        print(f"   Reduction: {899 - len(document.visual_elements)} duplicates removed")
        print(f"   Reduction rate: {(899 - len(document.visual_elements)) / 899 * 100:.1f}%")
    else:
        print(f"\n⚠️ No deduplication detected - still have {len(document.visual_elements)} visual elements")


if __name__ == "__main__":
    asyncio.run(test_deduplication())