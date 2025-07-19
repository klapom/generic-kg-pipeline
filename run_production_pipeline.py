#!/usr/bin/env python3
"""
Run the production pipeline with a BMW document and generate HTML report
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.parsers.parser_factory import ParserFactory
from core.pipeline_debugger_enhanced import (
    EnhancedPipelineDebugger, PipelineDebugConfig, DebugLevel
)
from tests.test_qwen25_html_report import generate_html_report, extract_page_images, extract_embedded_images_directly

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_pipeline(pdf_path: Path, output_dir: Path):
    """Run the production pipeline with enhanced debugging"""
    
    logger.info(f"🚀 Starting production pipeline for: {pdf_path.name}")
    
    # Configure enhanced debugging
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel.FULL,
        output_dir=output_dir,
        include_images=True,
        save_intermediate_results=True
    )
    
    # Initialize debugger
    debugger = EnhancedPipelineDebugger(debug_config)
    debugger.start_document_processing(pdf_path, pdf_path.stem)
    
    # Configure parser factory with standard parser
    parser_config = {
        'use_qwen25_parser': False,  # Use standard parser with Qwen2.5-VL for images
        'max_pages': 10,
        'pdfplumber_mode': 1,  # Fallback mode
        'enable_page_context': True,
        'page_context_pages': 5,
        'environment': 'production',
        'vlm': {
            'temperature': 0.2,
            'max_new_tokens': 512,
            'batch_size': 2,
            'enable_structured_parsing': True
        },
        'image_extraction': {
            'min_size': 100,
            'extract_embedded': True,
            'render_fallback': True
        }
    }
    
    # Create parser factory
    factory = ParserFactory(config=parser_config, enable_vlm=True)
    
    # Parse document
    logger.info("📄 Parsing document...")
    parser = factory.get_parser_for_file(pdf_path)
    document = await parser.parse(pdf_path)
    
    logger.info(f"✅ Parsing completed:")
    logger.info(f"   - Pages: {document.metadata.page_count}")
    logger.info(f"   - Segments: {len(document.segments)}")
    logger.info(f"   - Visual elements: {len(document.visual_elements)}")
    
    # Count segment types
    segment_types = {}
    for segment in document.segments:
        seg_type = f"{segment.segment_type.value}/{segment.segment_subtype or 'none'}"
        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
    
    logger.info("📊 Segment types:")
    for seg_type, count in segment_types.items():
        logger.info(f"   - {seg_type}: {count}")
    
    # Check for tables and charts
    tables = [s for s in document.segments if s.segment_type.value == 'table']
    charts = [s for s in document.segments if s.segment_type.value == 'visual' and s.segment_subtype == 'chart']
    
    if tables:
        logger.info(f"📊 Found {len(tables)} tables with structured data")
        for table in tables:
            if 'triple_count' in table.metadata:
                logger.info(f"   - Table on page {table.page_number}: {table.metadata['triple_count']} triples")
    
    if charts:
        logger.info(f"📈 Found {len(charts)} chart references")
    
    # Extract page images for report
    logger.info("🖼️ Extracting page images...")
    page_images = extract_page_images(pdf_path, max_pages=10)
    
    # If no visual elements found, try direct extraction
    if len(document.visual_elements) == 0:
        logger.info("🔍 No visual elements found, extracting images directly...")
        embedded_visuals = extract_embedded_images_directly(pdf_path, max_pages=10)
        if embedded_visuals:
            logger.info(f"   - Found {len(embedded_visuals)} embedded images")
            
            # Process with VLM if available
            if hasattr(parser, 'vlm_processor') and parser.vlm_processor:
                logger.info("🤖 Processing images with VLM...")
                results = await parser.vlm_processor.process_visual_elements(embedded_visuals)
                
                # Update visual elements with results
                for ve, result in zip(embedded_visuals, results):
                    if result.success:
                        ve.vlm_description = result.description
                        ve.confidence_score = result.confidence
                        if result.structured_data:
                            ve.analysis_metadata['structured_data'] = result.structured_data
                
                # Add to document
                document.visual_elements.extend(embedded_visuals)
    
    # Generate HTML report
    logger.info("📝 Generating HTML report...")
    report_path = generate_html_report(
        pdf_path=pdf_path,
        document=document,
        page_images=page_images,
        page_contexts=None,  # Could add page context if available
        output_path=Path(f"production_report_{pdf_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.html")
    )
    
    logger.info(f"✅ Report generated: {report_path}")
    
    # Cleanup
    if hasattr(parser, 'cleanup'):
        parser.cleanup()
    
    return document, report_path


async def main():
    """Main entry point"""
    # Select BMW document
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Run pipeline
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    document, report_path = await run_pipeline(pdf_path, output_dir)
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📄 Document: {pdf_path.name}")
    print(f"📝 Report: {report_path}")
    print(f"\nOpen the HTML report in your browser to view the results.")


if __name__ == "__main__":
    asyncio.run(main())