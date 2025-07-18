"""Example usage of vLLM SmolDocling client for PDF parsing"""

import asyncio
import logging
from pathlib import Path

from core.clients import VLLMSmolDoclingFinalClient as VLLMSmolDoclingClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def parse_single_pdf_example():
    """Example: Parse a single PDF with vLLM SmolDocling"""
    print("=== Single PDF Parsing Example ===")
    
    # Configuration is now passed directly to the client
    # The client handles all necessary settings internally
    
    # Path to your PDF file
    pdf_path = Path("path/to/your/document.pdf")
    
    if not pdf_path.exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        print("   Please update the path to point to an actual PDF file")
        return
    
    try:
        # Create client instance
        client = VLLMSmolDoclingClient()
        
        print(f"📄 Parsing PDF: {pdf_path.name}")
        
        # Parse the PDF
        result = client.parse_pdf(pdf_path)
            
            if result.success:
                print(f"✅ Parsing successful!")
                print(f"   📊 Pages processed: {result.total_pages}")
                print(f"   ⏱️  Processing time: {result.processing_time_seconds:.2f}s")
                print(f"   🤖 Model version: {result.model_version}")
                
                # Show summary of extracted content
                total_tables = sum(len(page.tables) for page in result.pages)
                total_images = sum(len(page.images) for page in result.pages)
                total_formulas = sum(len(page.formulas) for page in result.pages)
                
                print(f"   📊 Extracted content:")
                print(f"      - Tables: {total_tables}")
                print(f"      - Images: {total_images}")
                print(f"      - Formulas: {total_formulas}")
                
                # Convert to standard Document format
                document = client.convert_to_document(result, pdf_path)
                
                print(f"   📝 Document segments: {len(document.segments)}")
                print(f"   📏 Total content length: {len(document.content)} characters")
                
                # Show first few segments
                print("\n📋 First few segments:")
                for i, segment in enumerate(document.segments[:3]):
                    print(f"   {i+1}. [{segment.segment_type}] {segment.content[:100]}...")
                
            else:
                print(f"❌ Parsing failed: {result.error_message}")
                
    except Exception as e:
        print(f"❌ Error: {e}")


async def parse_multiple_pdfs_example():
    """Example: Batch parse multiple PDFs"""
    print("\n=== Batch PDF Parsing Example ===")
    
    # List of PDF files to process
    pdf_files = [
        Path("path/to/document1.pdf"),
        Path("path/to/document2.pdf"),
        Path("path/to/document3.pdf"),
    ]
    
    # Filter to existing files
    existing_files = [f for f in pdf_files if f.exists()]
    
    if not existing_files:
        print("⚠️  No PDF files found. Please update the paths to point to actual PDF files")
        return
    
    try:
        async with VLLMSmolDoclingClient() as client:
            print(f"📄 Batch parsing {len(existing_files)} PDFs...")
            
            # Parse all PDFs in batch
            results = await client.batch_parse_pdfs(existing_files)
            
            # Summary
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            total_time = sum(r.processing_time_seconds for r in results)
            
            print(f"✅ Batch parsing completed!")
            print(f"   📊 Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   📈 Average time per document: {total_time/len(results):.2f}s")
            
            # Show details for each file
            for pdf_file, result in zip(existing_files, results):
                if result.success:
                    print(f"   ✅ {pdf_file.name}: {result.total_pages} pages, "
                          f"{result.processing_time_seconds:.2f}s")
                else:
                    print(f"   ❌ {pdf_file.name}: {result.error_message}")
                    
    except Exception as e:
        print(f"❌ Batch parsing error: {e}")


async def health_check_example():
    """Example: Check vLLM SmolDocling service health"""
    print("\n=== Health Check Example ===")
    
    try:
        async with VLLMSmolDoclingClient() as client:
            health = await client.health_check()
            
            print(f"🏥 Service status: {health['status']}")
            print(f"🔗 Endpoint: {health['endpoint']}")
            
            if health['status'] == 'healthy':
                print(f"⚡ Response time: {health.get('response_time_ms', 0):.2f}ms")
                
                model_info = health.get('model_info', {})
                if model_info:
                    print(f"🤖 Model info: {model_info}")
                
                gpu_info = health.get('gpu_info', {})
                if gpu_info:
                    print(f"🖥️  GPU info: {gpu_info}")
            else:
                print(f"❌ Error: {health.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")


async def advanced_configuration_example():
    """Example: Advanced configuration options"""
    print("\n=== Advanced Configuration Example ===")
    
    # Different configurations for different use cases
    configs = {
        "fast_preview": SmolDoclingConfig(
            max_pages=5,
            extract_tables=False,
            extract_images=False,
            extract_formulas=False,
            preserve_layout=False,
            timeout_seconds=30
        ),
        "full_extraction": SmolDoclingConfig(
            max_pages=100,
            extract_tables=True,
            extract_images=True,
            extract_formulas=True,
            preserve_layout=True,
            timeout_seconds=300
        ),
        "table_focused": SmolDoclingConfig(
            max_pages=50,
            extract_tables=True,
            extract_images=False,
            extract_formulas=False,
            preserve_layout=True,
            timeout_seconds=180
        )
    }
    
    for config_name, config in configs.items():
        print(f"\n📋 Configuration: {config_name}")
        print(f"   Max pages: {config.max_pages}")
        print(f"   Extract tables: {config.extract_tables}")
        print(f"   Extract images: {config.extract_images}")
        print(f"   Extract formulas: {config.extract_formulas}")
        print(f"   Preserve layout: {config.preserve_layout}")
        print(f"   Timeout: {config.timeout_seconds}s")


async def main():
    """Run all examples"""
    print("🚀 vLLM SmolDocling Client Examples")
    print("=" * 50)
    
    # Run examples
    await health_check_example()
    await advanced_configuration_example()
    await parse_single_pdf_example()
    await parse_multiple_pdfs_example()
    
    print("\n✨ Examples completed!")
    print("\n💡 To use with real PDFs:")
    print("   1. Update the PDF file paths in the examples")
    print("   2. Make sure vLLM SmolDocling service is running")
    print("   3. Configure the endpoint in config/default.yaml")


if __name__ == "__main__":
    asyncio.run(main())