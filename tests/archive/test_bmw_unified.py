#!/usr/bin/env python3
"""
Test BMW Document Processing with Unified Architecture
Shows the complete flow with new VLLMSmolDoclingFinalClient
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"bmw_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure both file and console logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set specific loggers to DEBUG
logging.getLogger('core.clients.vllm_smoldocling_final').setLevel(logging.DEBUG)

# Import components
from core.parsers.implementations.pdf.hybrid_pdf_parser import HybridPDFParser
from core.content_chunker import ContentChunker
from core.clients.hochschul_llm import HochschulLLMClient, TripleExtractionConfig

async def test_bmw_document():
    """Test complete pipeline with BMW document"""
    
    print("\n" + "="*80)
    print("🚗 BMW Document Processing Test - Unified Architecture")
    print("="*80 + "\n")
    
    # Select document
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"📄 Testing with: {test_file.name}")
    print(f"   File size: {test_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Show current configuration
    print("\n🔧 Configuration:")
    print(f"   USE_DOCLING: {os.getenv('USE_DOCLING', 'True')}")
    print(f"   EXTRACT_IMAGES_DIRECTLY: {os.getenv('EXTRACT_IMAGES_DIRECTLY', 'True')}")
    print(f"   MAX_IMAGE_SIZE: {os.getenv('MAX_IMAGE_SIZE', '2048')}")
    print(f"   Environment: production (docling/SmolDocling enabled)")
    print(f"\n📝 Logging to: {log_file}")
    
    try:
        # Step 1: Parse document
        print("\n📋 Step 1: Parsing Document")
        parser = HybridPDFParser(
            config={
                'environment': 'production',  # This enables docling/SmolDocling
                'max_pages': 3,  # Limit for debugging
                'gpu_memory_utilization': 0.3
            },
            enable_vlm=True  # Enable visual analysis to test full functionality
        )
        
        print(f"   Parser initialized with client: {type(parser.smoldocling_client).__name__}")
        
        start_time = datetime.now()
        document = await parser.parse(test_file)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Parsing completed in {parse_time:.2f}s")
        print(f"   Total segments: {document.total_segments}")
        print(f"   Total visual elements: {document.total_visual_elements}")
        
        # Show segment content
        if document.segments:
            print(f"\n📄 Document Segments:")
            for i, seg in enumerate(document.segments[:5]):
                text_preview = seg.content[:100] if seg.content else "No content"
                seg_type = seg.segment_type.value if hasattr(seg.segment_type, 'value') else str(seg.segment_type)
                print(f"   [{i}] {seg_type}: {text_preview}...")
        
        # Show visual elements
        if document.visual_elements:
            print(f"\n🖼️  Visual Elements:")
            for i, vis in enumerate(document.visual_elements[:3]):
                print(f"   [{i}] {vis.element_type.value}: {vis.vlm_description[:100] if vis.vlm_description else 'No description'}...")
        
        # Step 2: Chunking
        print("\n📦 Step 2: Chunking Document")
        chunker = ContentChunker()
        chunking_result = await chunker.chunk_document(document)
        chunks = chunking_result.contextual_chunks if hasattr(chunking_result, 'contextual_chunks') else []
        
        print(f"✅ Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            chunk_text_len = len(chunk.text) if hasattr(chunk, 'text') else len(chunk.content) if hasattr(chunk, 'content') else 0
            visuals = len(chunk.visual_elements) if hasattr(chunk, 'visual_elements') else 0
            print(f"   Chunk {i}: {chunk_text_len} chars, {visuals} visuals")
        
        # Step 3: Triple Extraction (if Hochschul-LLM available)
        try:
            print("\n🔗 Step 3: Triple Extraction")
            llm_client = HochschulLLMClient()
            
            # Test with first chunk
            if chunks:
                config = TripleExtractionConfig(
                    extract_entities=True,
                    extract_relationships=True,
                    use_ontology=False
                )
                
                chunk_text = chunks[0].content if hasattr(chunks[0], 'content') else chunks[0].text
                result = await llm_client.extract_triples(
                    text=chunk_text[:1000],  # Limit text for testing
                    config=config
                )
                
                print(f"✅ Extracted {len(result.triples)} triples")
                for i, triple in enumerate(result.triples[:5]):
                    print(f"   [{i}] {triple.subject} --{triple.predicate}--> {triple.object}")
                    
        except Exception as e:
            print(f"⚠️  Triple extraction skipped: {e}")
        
        # Summary
        print("\n📊 Processing Summary:")
        print(f"   ✅ Document parsed successfully")
        print(f"   ✅ Client: VLLMSmolDoclingFinalClient")
        print(f"   ✅ Docling integration: {'Enabled' if parser.smoldocling_client.use_docling else 'Disabled'}")
        print(f"   ✅ Total processing time: {parse_time:.2f}s")
        
        # Save sample output
        output_dir = Path("data/test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"bmw_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'document': test_file.name,
                'segments': document.total_segments,
                'visual_elements': document.total_visual_elements,
                'chunks': len(chunks),
                'processing_time': parse_time,
                'client': type(parser.smoldocling_client).__name__,
                'docling_enabled': parser.smoldocling_client.use_docling,
                'environment': parser.smoldocling_client.environment
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")

def main():
    """Run the test"""
    # Set environment for production (enables docling)
    os.environ['USE_DOCLING'] = 'true'
    os.environ['EXTRACT_IMAGES_DIRECTLY'] = 'true'
    os.environ['LOG_PERFORMANCE'] = 'true'
    
    asyncio.run(test_bmw_document())

if __name__ == "__main__":
    main()