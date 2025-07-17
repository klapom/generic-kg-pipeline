#!/usr/bin/env python3
"""
Test script to verify SmolDocling content extraction is working after fix
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.parsers.implementations.pdf import HybridPDFParser


async def test_content_extraction():
    """Test that content is properly extracted from PDFs"""
    
    # Look for a test PDF
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/input/")
        return
    
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    print("=" * 60)
    
    # Parse with HybridPDFParser
    parser = HybridPDFParser(enable_vlm=False)
    
    try:
        document = await parser.parse(test_pdf)
        
        print(f"✅ Parsing successful!")
        print(f"   - Total segments: {len(document.segments)}")
        print(f"   - Total visual elements: {len(document.visual_elements)}")
        print()
        
        # Check for error messages
        error_segments = [s for s in document.segments if "[Error processing page]" in s.content]
        
        if error_segments:
            print(f"❌ Found {len(error_segments)} segments with errors!")
        else:
            print(f"✅ No error segments found!")
        
        # Show content from first few segments
        print("\nFirst 3 segments:")
        print("-" * 60)
        
        for i, segment in enumerate(document.segments[:3]):
            print(f"\nSegment {i+1} (Page {segment.page_number}):")
            print(f"Type: {segment.segment_type}")
            print(f"Content preview (first 200 chars):")
            print(segment.content[:200])
            if len(segment.content) > 200:
                print("...")
            print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_content_extraction())