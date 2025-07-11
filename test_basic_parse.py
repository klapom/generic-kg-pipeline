#!/usr/bin/env python3
"""
Basic test script to parse BMW 3er G20 PDF without vLLM
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from plugins.parsers.pdf_parser import PDFParser
from plugins.parsers.base_parser import DocumentType

def test_basic_pdf_parse():
    """Test basic PDF parsing without vLLM"""
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"🔍 Testing basic PDF parse: {pdf_path.name}")
    
    try:
        # Create PDF parser without VLM
        parser = PDFParser(enable_vlm=False)
        
        # Parse the PDF
        print("📄 Parsing PDF...")
        document = parser.parse(pdf_path)
        
        # Print results
        print(f"✅ Successfully parsed PDF!")
        print(f"📊 Document statistics:")
        print(f"   - Title: {document.metadata.title}")
        print(f"   - Pages: {document.metadata.page_count}")
        print(f"   - Content length: {len(document.content)} chars")
        print(f"   - Segments: {len(document.segments)}")
        
        # Show first segment
        if document.segments:
            first_segment = document.segments[0]
            print(f"   - First segment type: {first_segment.segment_type}")
            print(f"   - First segment content: {first_segment.content[:200]}...")
        
        # Show document type
        print(f"   - Document type: {document.document_type}")
        
        return document
        
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_basic_pdf_parse()