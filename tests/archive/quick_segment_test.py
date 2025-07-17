#!/usr/bin/env python3
"""
Quick segment comparison test - process just first few pages
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
import base64
from pdf2image import convert_from_path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.content_chunker import ContentChunker

async def quick_segment_test():
    """Process first few pages for quick testing"""
    
    # Process just BMW X5
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"📄 Processing first 3 pages of: {pdf_path.name}")
    
    # Parse with SmolDocling
    parser = HybridPDFParser(enable_vlm=False)
    
    # Convert first 3 pages to images
    print("📸 Converting PDF pages to images...")
    images = convert_from_path(str(pdf_path), first_page=1, last_page=3, dpi=150)
    
    # Parse document
    print("🤖 Parsing with SmolDocling...")
    document = await parser.parse(pdf_path)
    
    # Filter segments for first 3 pages
    page_segments = [s for s in document.segments if s.page_number and s.page_number <= 3]
    
    print(f"✅ Found {len(page_segments)} segments in first 3 pages")
    
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
    
    # Create a temporary document with only first 3 pages segments
    temp_doc = document.model_copy()
    temp_doc.segments = page_segments
    
    chunking_result = await chunker.chunk_document(temp_doc)
    
    print(f"📦 Created {len(chunking_result.chunks)} chunks")
    
    # Generate HTML
    output_dir = Path("tests/debugging/segment_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW X5 - Segment Comparison (First 3 Pages)</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        .page-comparison {
            background: white;
            margin-bottom: 3rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .page-header {
            background: #f8f9fa;
            padding: 1rem 2rem;
            border-bottom: 1px solid #e9ecef;
            font-weight: bold;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 400px 1fr 1fr;
            gap: 2rem;
            padding: 2rem;
        }
        .column {
            overflow: auto;
        }
        .column-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
        .pdf-image {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .segment {
            background: #f8f9fa;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }
        .segment-type {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        .chunk {
            background: #e9ecef;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #764ba2;
        }
        .chunk-meta {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .content-text {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 BMW X5 - Segment Comparison</h1>
        <p>SmolDocling PDF Processing → Segments → Contextual Chunks</p>
        <p style="opacity: 0.8;">First 3 Pages Preview</p>
    </div>
    
    <div class="container">
"""
    
    for page_num in range(1, 4):
        # Convert image to base64
        img_buffer = Path(f"/tmp/page_{page_num}.png")
        images[page_num-1].save(img_buffer, "PNG")
        with open(img_buffer, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        # Get segments for this page
        page_segs = [s for s in page_segments if s.page_number == page_num]
        
        # Get chunks that reference this page
        page_chunks = [c for c in chunking_result.chunks 
                      if any(page_num in s.page_numbers for s in c.source_segments)]
        
        html_content += f"""
        <div class="page-comparison">
            <div class="page-header">📄 Page {page_num}</div>
            <div class="comparison-grid">
                <div class="column">
                    <div class="column-title">Original PDF</div>
                    <img src="data:image/png;base64,{img_base64}" class="pdf-image" alt="Page {page_num}">
                </div>
                
                <div class="column">
                    <div class="column-title">Extracted Segments ({len(page_segs)})</div>
"""
        
        for seg in page_segs:
            html_content += f"""
                    <div class="segment">
                        <div class="segment-type">Type: {seg.segment_type}</div>
                        <div class="content-text">{seg.content[:300]}{"..." if len(seg.content) > 300 else ""}</div>
                    </div>
"""
        
        html_content += """
                </div>
                
                <div class="column">
                    <div class="column-title">Contextual Chunks (""" + str(len(page_chunks)) + """)</div>
"""
        
        for chunk in page_chunks:
            html_content += f"""
                    <div class="chunk">
                        <div class="content-text">{chunk.content[:300]}{"..." if len(chunk.content) > 300 else ""}</div>
                        <div class="chunk-meta">
                            Tokens: {chunk.token_count} | Type: {chunk.chunk_type.value}
                        </div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"bmw_x5_segment_comparison_{timestamp}.html"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Segment comparison saved to: {output_file}")
    print(f"📂 Location: tests/debugging/segment_comparison/")
    
    # Clean up temp images
    for i in range(1, 4):
        Path(f"/tmp/page_{i}.png").unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(quick_segment_test())