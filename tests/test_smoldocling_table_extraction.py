#!/usr/bin/env python3
"""
Test SmolDocling table extraction directly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
from core.parsers.interfaces import DocumentType
import asyncio


async def test_table_extraction():
    """Test how SmolDocling extracts tables"""
    
    # Initialize client
    client = VLLMSmolDoclingFinalClient()
    
    # Process test PDF
    pdf_path = Path("data/input/test_tables_charts.pdf")
    if not pdf_path.exists():
        print(f"Test PDF not found: {pdf_path}")
        return
    
    print(f"Processing: {pdf_path}")
    
    # Process document
    result = await client.process_document(pdf_path, doc_type=DocumentType.PDF)
    
    # Check for table segments
    table_segments = []
    for segment in result.segments:
        # Check if segment contains table
        if 'table' in segment.segment_type.value.lower() or (segment.segment_subtype and 'table' in segment.segment_subtype.lower()):
            table_segments.append(segment)
        elif 'TableCell' in segment.content or '|' in segment.content:
            # Also check content for table indicators
            table_segments.append(segment)
    
    print(f"\nFound {len(table_segments)} segments with table content")
    
    for i, segment in enumerate(table_segments[:3]):  # Show first 3
        print(f"\n--- Table Segment {i+1} ---")
        print(f"Type: {segment.segment_type.value}")
        print(f"Subtype: {segment.segment_subtype}")
        print(f"Page: {segment.page_number}")
        print(f"Content preview:")
        # Show first 500 chars
        content_preview = segment.content[:500]
        print(content_preview)
        if len(segment.content) > 500:
            print("... (truncated)")
        
        # Check if it's formatted as Markdown table
        if '|' in content_preview and '---' in content_preview:
            print("✅ This looks like a properly formatted Markdown table!")
        elif 'TableCell' in content_preview:
            print("❌ This is still in TableCell format - fix may not be applied")
    
    # Cleanup
    client.cleanup()


if __name__ == "__main__":
    asyncio.run(test_table_extraction())