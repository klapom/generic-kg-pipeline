#!/usr/bin/env python3
"""
Process single BMW document with SmolDocling and generate comparison
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.content_chunker import ContentChunker

async def process_single_bmw():
    """Process single BMW document"""
    
    # Find BMW X5 document
    input_dir = Path("data/input")
    bmw_x5 = input_dir / "Preview_BMW_X5_G05.pdf"
    
    if not bmw_x5.exists():
        print(f"❌ File not found: {bmw_x5}")
        return
    
    print(f"📄 Processing: {bmw_x5.name}")
    
    # Parse with SmolDocling
    parser = HybridPDFParser(enable_vlm=False)
    document = await parser.parse(bmw_x5)
    
    print(f"✅ Parsed document:")
    print(f"   📊 Total segments: {len(document.segments)}")
    print(f"   🖼️ Visual elements: {len(document.visual_elements)}")
    print(f"   📄 Pages: {document.metadata.page_count}")
    
    # Create chunks
    chunker = ContentChunker({
        "chunking": {
            "strategies": {
                "pdf": {
                    "max_tokens": 500,
                    "min_tokens": 100,
                    "overlap_tokens": 50,
                    "respect_boundaries": True
                }
            }
        }
    })
    
    chunking_result = await chunker.chunk_document(document)
    
    print(f"✅ Created chunks:")
    print(f"   📦 Total chunks: {len(chunking_result.chunks)}")
    
    # Show first few segments and chunks
    print("\n📝 First 3 segments:")
    for i, segment in enumerate(document.segments[:3]):
        print(f"   {i+1}. [{segment.segment_type}] {segment.content[:100]}...")
    
    print("\n📦 First 3 chunks:")
    for i, chunk in enumerate(chunking_result.chunks[:3]):
        print(f"   {i+1}. [{chunk.chunk_type.value}] {chunk.content[:100]}...")
    
    # Save results
    output_dir = Path("tests/debugging/segment_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "document_name": bmw_x5.name,
        "stats": {
            "total_segments": len(document.segments),
            "total_visual_elements": len(document.visual_elements),
            "total_chunks": len(chunking_result.chunks),
            "pages": document.metadata.page_count
        },
        "segments": [
            {
                "page": s.page_number,
                "type": s.segment_type,
                "content": s.content[:200]
            }
            for s in document.segments
        ],
        "chunks": [
            {
                "id": c.chunk_id,
                "type": c.chunk_type.value,
                "tokens": c.token_count,
                "content": c.content[:200]
            }
            for c in chunking_result.chunks
        ]
    }
    
    output_file = output_dir / f"{bmw_x5.stem}_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎯 SmolDocling Processing Complete!")

if __name__ == "__main__":
    asyncio.run(process_single_bmw())