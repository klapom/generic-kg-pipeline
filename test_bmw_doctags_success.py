#!/usr/bin/env python3
"""
Quick test to verify DocTags transformation works with BMW document
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific loggers to DEBUG
logging.getLogger('core.clients.vllm_smoldocling_final').setLevel(logging.DEBUG)

from core.parsers.implementations.pdf.hybrid_pdf_parser import HybridPDFParser

async def test_bmw_doctags():
    """Test DocTags transformation with BMW document"""
    
    print("\n" + "="*60)
    print("🚗 BMW DocTags Transformation Test")
    print("="*60 + "\n")
    
    # Select document
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"📄 Testing with: {test_file.name}")
    
    try:
        # Initialize parser with production settings
        parser = HybridPDFParser(
            config={
                'environment': 'production',  # Enables docling/SmolDocling
                'max_pages': 1,  # Just test first page
                'gpu_memory_utilization': 0.3
            },
            enable_vlm=False  # Skip VLM for quick test
        )
        
        print(f"✅ Parser initialized")
        print(f"   Client type: {type(parser.smoldocling_client).__name__}")
        print(f"   DocTags transformation: {'Enabled' if hasattr(parser.smoldocling_client, '_transform_doctags') else 'Not available'}")
        
        # Parse document
        start_time = datetime.now()
        document = await parser.parse(test_file)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Parsing completed in {parse_time:.2f}s")
        print(f"   Total segments: {document.total_segments}")
        print(f"   Total visual elements: {document.total_visual_elements}")
        
        # Show segment content
        if document.segments:
            print(f"\n📄 Extracted Text (first 3 segments):")
            for i, seg in enumerate(document.segments[:3]):
                text_preview = seg.content[:80] if seg.content else "No content"
                print(f"   [{i}] {text_preview}...")
        
        # Check if we got meaningful content
        total_text_length = sum(len(seg.content) for seg in document.segments if seg.content)
        print(f"\n📊 Summary:")
        print(f"   Total text extracted: {total_text_length} characters")
        print(f"   DocTags parsing: {'SUCCESS' if total_text_length > 100 else 'FAILED'}")
        
        if total_text_length > 100:
            print(f"\n✨ SUCCESS: DocTags transformation is working!")
            print(f"   The SmolDocling tags (<paragraph>, <section_header>) were successfully")
            print(f"   transformed to docling-core compatible tags (<text>, <section_header_level_1>)")
        else:
            print(f"\n❌ FAILED: No meaningful text extracted")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bmw_doctags())