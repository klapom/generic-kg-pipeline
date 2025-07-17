#!/usr/bin/env python3
"""
Quick Debug Pipeline Test

A faster version that uses mock data for demonstration of the debug pipeline.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline_debugger import PipelineDebugConfig, DebugLevel, PipelineDebugger
from core.parsers import Document, DocumentMetadata, DocumentType, Segment, VisualElement
from core.parsers.interfaces.data_models import VisualElementType
from core.content_chunker import ContentChunker, ContextualChunk, ChunkingResult


async def create_mock_document(pdf_path: Path) -> Document:
    """Create a mock document for testing"""
    
    # Create metadata
    metadata = DocumentMetadata(
        title=pdf_path.stem,
        document_type=DocumentType.PDF,
        page_count=5,
        file_path=pdf_path,
        created_date=datetime.now()
    )
    
    # Create segments
    segments = []
    for page in range(1, 6):
        # Text segment
        segments.append(Segment(
            content=f"This is the main text content from page {page} of {pdf_path.name}. "
                   f"It contains important information about BMW vehicles including "
                   f"technical specifications, features, and design elements.",
            page_number=page,
            segment_index=len(segments),
            segment_type="text"
        ))
        
        # Add visual element on some pages
        if page in [2, 4]:
            visual_id = f"hash_{page}"
            segments.append(Segment(
                content=f"[Image placeholder for page {page}]",
                page_number=page,
                segment_index=len(segments),
                segment_type="image",
                visual_references=[visual_id]
            ))
    
    # Create visual elements
    visual_elements = []
    for page in [2, 4]:
        visual_elements.append(VisualElement(
            element_type=VisualElementType.DIAGRAM if page == 2 else VisualElementType.IMAGE,
            source_format=DocumentType.PDF,
            content_hash=f"hash_{page}",
            page_or_slide=page,
            confidence=0.95,
            raw_data=b"mock_image_data"  # Mock image data
        ))
    
    # Create document
    document = Document(
        document_id=f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        source_path=str(pdf_path),
        document_type=DocumentType.PDF,
        content="\n\n".join([s.content for s in segments]),
        metadata=metadata,
        segments=segments,
        visual_elements=visual_elements
    )
    
    return document


async def test_quick_debug():
    """Run a quick debug test with mock data"""
    
    print("🚀 Quick Debug Pipeline Test")
    print("=" * 60)
    
    # Create debug configuration
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel.DETAILED,
        generate_html_report=True,
        track_segments=True,
        track_chunks=True,
        track_vlm_descriptions=True,
        save_intermediate_results=True,
        output_dir=Path("data/debug/quick_test")
    )
    
    # Initialize debugger
    debugger = PipelineDebugger(debug_config)
    
    # Test file
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    doc_id = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📄 Creating mock document for: {pdf_path.name}")
    print(f"🔍 Debug Level: {debug_config.debug_level.value}")
    print()
    
    # Start debugging
    debugger.start_document_processing(pdf_path, doc_id)
    
    try:
        # STAGE 1: Mock parsing
        print("📄 Stage 1: Mock Parsing")
        import time
        start_time = time.time()
        
        document = await create_mock_document(pdf_path)
        
        parsing_time = time.time() - start_time
        debugger.track_parsing_complete(document, parsing_time)
        
        print(f"   ✅ Created mock document: {len(document.segments)} segments, {len(document.visual_elements)} visual elements")
        
        # STAGE 2: Mock VLM Processing
        if document.visual_elements:
            print("🤖 Stage 2: Mock VLM Processing")
            start_time = time.time()
            
            for visual in document.visual_elements:
                # Simulate VLM processing
                processing_time = 0.5  # Mock processing time
                
                description = f"This is a detailed VLM analysis of the {visual.element_type.value} on page {visual.page_or_slide}. " \
                             f"The image shows technical diagrams and specifications related to BMW vehicle systems."
                
                confidence = 0.85 if visual.element_type == VisualElementType.DIAGRAM else 0.92
                
                debugger.track_vlm_processing(
                    segment_id=visual.content_hash,
                    model="qwen2.5-vl" if visual.element_type == VisualElementType.IMAGE else "pixtral",
                    description=description,
                    confidence=confidence,
                    processing_time=processing_time
                )
                
                # Update segment
                segment = next((s for s in document.segments if visual.content_hash in s.visual_references), None)
                if segment:
                    segment.content = f"{segment.content}\n\n[VLM Description: {description}]"
            
            total_vlm_time = time.time() - start_time
            print(f"   ✅ VLM processed {len(document.visual_elements)} visual elements")
        
        # STAGE 3: Chunking
        print("📦 Stage 3: Content Chunking")
        start_time = time.time()
        
        chunking_config = {
            "chunking": {
                "strategies": {
                    "pdf": {
                        "max_tokens": 300,
                        "min_tokens": 50,
                        "overlap_tokens": 30,
                        "respect_boundaries": True
                    }
                }
            }
        }
        
        chunker = ContentChunker(chunking_config)
        chunking_result = await chunker.chunk_document(document)
        
        chunking_time = time.time() - start_time
        debugger.track_chunking_complete(chunking_result, chunking_time)
        
        print(f"   ✅ Created {len(chunking_result.contextual_chunks)} chunks")
        
        # Add some warnings for demonstration
        debugger.track_warning("vlm_processing", "Low confidence detected for some visual elements")
        
        # End debugging and generate report
        report_path = debugger.end_document_processing()
        
        print(f"\n✅ Debug test complete!")
        if report_path:
            print(f"📊 Debug report generated:")
            print(f"   {report_path}")
            print(f"\n   Open this file in your browser to see:")
            print(f"   - Pipeline flow visualization")
            print(f"   - Segment analysis with VLM descriptions")
            print(f"   - Chunk formation details")
            print(f"   - Performance metrics")
        
    except Exception as e:
        debugger.track_error("processing", e)
        debugger.end_document_processing()
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_quick_debug())