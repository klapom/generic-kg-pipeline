#!/usr/bin/env python3
"""
Test the Debug Pipeline with BMW Documents

Demonstrates the production pipeline with full debugging and HTML report generation.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.process_documents_debug import DebugDocumentProcessor
from core.pipeline_debugger import PipelineDebugConfig, DebugLevel


async def test_debug_pipeline():
    """Test the debug pipeline with a BMW document"""
    
    print("🚀 Testing Debug Pipeline with BMW Document")
    print("=" * 60)
    
    # Create debug configuration with full tracking
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel.DETAILED,
        generate_html_report=True,
        track_segments=True,
        track_chunks=True,
        track_vlm_descriptions=True,
        save_intermediate_results=True,
        output_dir=Path("data/debug"),
        include_images=False,  # Set to True for full image inclusion
        max_content_preview=500
    )
    
    # Initialize processor
    processor = DebugDocumentProcessor(debug_config)
    
    # Find BMW X5 document
    input_dir = Path("data/input")
    bmw_x5 = input_dir / "Preview_BMW_X5_G05.pdf"
    
    if not bmw_x5.exists():
        print(f"❌ File not found: {bmw_x5}")
        print("Please ensure BMW documents are in data/input/")
        return
    
    print(f"📄 Processing: {bmw_x5.name}")
    print(f"🔍 Debug Level: {debug_config.debug_level.value}")
    print(f"📊 HTML Report: Enabled")
    print(f"💾 Intermediate Results: Enabled")
    print()
    
    try:
        # Process the document
        result = await processor.process_file_with_debug(bmw_x5, enable_vlm=True)
        
        print("\n✅ Processing Complete!")
        print(f"📊 Summary:")
        print(f"   - Segments: {result['processing_summary']['total_segments']}")
        print(f"   - Visual Elements: {result['processing_summary']['total_visual_elements']}")
        print(f"   - Chunks: {result['processing_summary']['total_chunks']}")
        print(f"   - VLM Processed: {result['processing_summary']['vlm_processed']}")
        
        if result.get('debug_report'):
            print(f"\n📈 Debug Report Generated:")
            print(f"   {result['debug_report']}")
            print(f"\n   Open this file in your browser to see:")
            print(f"   - Complete pipeline flow visualization")
            print(f"   - Segment-by-segment analysis")
            print(f"   - VLM descriptions and confidence scores")
            print(f"   - Chunk formation and context inheritance")
            print(f"   - Performance metrics for each stage")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_batch_debug():
    """Test batch processing with debug"""
    
    print("\n🚀 Testing Batch Debug Pipeline")
    print("=" * 60)
    
    # Create debug configuration
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel.BASIC,
        generate_html_report=True,
        track_segments=True,
        track_chunks=True,
        track_vlm_descriptions=True,
        save_intermediate_results=False,
        output_dir=Path("data/debug/batch")
    )
    
    # Initialize processor
    processor = DebugDocumentProcessor(debug_config)
    
    # Find all BMW documents
    input_dir = Path("data/input")
    bmw_files = list(input_dir.glob("Preview_BMW*.pdf"))
    
    if not bmw_files:
        print("❌ No BMW documents found in data/input/")
        return
    
    print(f"📁 Found {len(bmw_files)} BMW documents")
    for f in bmw_files:
        print(f"   - {f.name}")
    
    print(f"\n🔍 Debug Level: {debug_config.debug_level.value}")
    print(f"📊 HTML Reports: Enabled")
    print()
    
    try:
        # Process all documents
        results = await processor.process_batch_with_debug(bmw_files, enable_vlm=True)
        
        print("\n✅ Batch Processing Complete!")
        print(f"📊 Summary:")
        print(f"   - Total Files: {len(results)}")
        print(f"   - Successful: {sum(1 for r in results if 'error' not in r)}")
        print(f"   - Failed: {sum(1 for r in results if 'error' in r)}")
        
        print(f"\n📈 Debug Reports Generated in: {debug_config.output_dir}")
        for result in results:
            if result.get('debug_report'):
                print(f"   - {Path(result['debug_report']).name}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    
    print("🔍 Production Pipeline Debug Test")
    print("This demonstrates the debug capabilities for production use")
    print()
    
    # Run single file test
    asyncio.run(test_debug_pipeline())
    
    # Ask if user wants to run batch test
    print("\n" + "="*60)
    response = input("\nRun batch processing test? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(test_batch_debug())


if __name__ == "__main__":
    main()