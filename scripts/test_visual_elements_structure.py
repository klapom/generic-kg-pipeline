#!/usr/bin/env python3
"""
Test script to examine visual element structure in the PDF parsing pipeline
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.parsers.interfaces import VisualElementType, DocumentType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_visual_element_details(visual_elem, index):
    """Print detailed information about a visual element"""
    print(f"\n{'='*60}")
    print(f"Visual Element #{index + 1}")
    print(f"{'='*60}")
    
    # Basic fields
    print(f"Element Type: {visual_elem.element_type.value}")
    print(f"Source Format: {visual_elem.source_format.value}")
    print(f"Content Hash: {visual_elem.content_hash}")
    print(f"Page/Slide: {visual_elem.page_or_slide}")
    print(f"File Extension: {visual_elem.file_extension}")
    print(f"Confidence: {visual_elem.confidence}")
    
    # VLM Description
    print(f"\nVLM Description: {'[FILLED]' if visual_elem.vlm_description else '[EMPTY]'}")
    if visual_elem.vlm_description:
        print(f"  Description: {visual_elem.vlm_description[:200]}{'...' if len(visual_elem.vlm_description) > 200 else ''}")
    
    # Bounding Box
    print(f"\nBounding Box: {'[PRESENT]' if visual_elem.bounding_box else '[MISSING]'}")
    if visual_elem.bounding_box:
        print(f"  Coordinates: {visual_elem.bounding_box}")
    
    # Extracted Data
    print(f"\nExtracted Data: {'[PRESENT]' if visual_elem.extracted_data else '[EMPTY]'}")
    if visual_elem.extracted_data:
        print(f"  Data: {json.dumps(visual_elem.extracted_data, indent=2)}")
    
    # Segment Reference
    print(f"\nSegment Reference: {visual_elem.segment_reference or '[NONE]'}")
    
    # Raw Data
    print(f"Raw Data: {'[PRESENT]' if visual_elem.raw_data else '[MISSING]'}")
    if visual_elem.raw_data:
        print(f"  Size: {len(visual_elem.raw_data)} bytes")
    
    # Analysis Metadata
    print(f"\nAnalysis Metadata: {'[PRESENT]' if visual_elem.analysis_metadata else '[EMPTY]'}")
    if visual_elem.analysis_metadata:
        print(f"  Metadata: {json.dumps(visual_elem.analysis_metadata, indent=2)}")


def print_segment_visual_references(document):
    """Print how segments reference visual elements"""
    print(f"\n\n{'='*60}")
    print("Segment-Visual Element Relationships")
    print(f"{'='*60}")
    
    for i, segment in enumerate(document.segments):
        if segment.visual_references:
            print(f"\nSegment #{i + 1} (Page {segment.page_number}):")
            print(f"  Type: {segment.segment_type}")
            print(f"  Visual References: {segment.visual_references}")
            print(f"  Content Preview: {segment.content[:100]}...")


async def test_pdf_visual_elements(pdf_path: Path):
    """Test PDF parsing and examine visual element structure"""
    print(f"\n{'#'*60}")
    print(f"Testing Visual Element Structure for: {pdf_path.name}")
    print(f"{'#'*60}")
    
    # Initialize parser
    config = {
        'extract_images': True,
        'extract_tables': True,
        'extract_formulas': True,
        'environment': 'test',  # Use test environment to avoid GPU issues
        'pdfplumber_mode': 1,   # Use fallback mode
        'max_pages': 5          # Limit pages for testing
    }
    
    # Test without VLM first to see raw structure
    print("\n1. Testing WITHOUT VLM analysis...")
    parser_no_vlm = HybridPDFParser(config=config, enable_vlm=False)
    
    try:
        # Parse document
        document = await parser_no_vlm.parse(pdf_path)
        
        print(f"\nDocument Summary:")
        print(f"  Total Segments: {document.total_segments}")
        print(f"  Total Visual Elements: {document.total_visual_elements}")
        print(f"  Visual Element Types: {document.visual_element_types}")
        
        # Print details of each visual element
        if document.visual_elements:
            for i, visual_elem in enumerate(document.visual_elements):
                print_visual_element_details(visual_elem, i)
        else:
            print("\nNo visual elements found!")
        
        # Print segment-visual relationships
        print_segment_visual_references(document)
        
    except Exception as e:
        logger.error(f"Error parsing PDF without VLM: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with VLM if available
    print("\n\n2. Testing WITH VLM analysis...")
    parser_with_vlm = HybridPDFParser(config=config, enable_vlm=True)
    
    try:
        # Parse document with VLM
        document_vlm = await parser_with_vlm.parse(pdf_path)
        
        print(f"\nDocument Summary (with VLM):")
        print(f"  Total Segments: {document_vlm.total_segments}")
        print(f"  Total Visual Elements: {document_vlm.total_visual_elements}")
        print(f"  VLM Analyzed Elements: {sum(1 for ve in document_vlm.visual_elements if ve.vlm_description)}")
        
        # Compare VLM descriptions
        if document_vlm.visual_elements:
            print("\nVLM Analysis Results:")
            for i, visual_elem in enumerate(document_vlm.visual_elements):
                print(f"\nVisual Element #{i + 1}:")
                print(f"  Type: {visual_elem.element_type.value}")
                print(f"  VLM Description: {'[FILLED]' if visual_elem.vlm_description else '[EMPTY]'}")
                if visual_elem.vlm_description:
                    print(f"    {visual_elem.vlm_description}")
                print(f"  Confidence: {visual_elem.confidence}")
        
    except Exception as e:
        logger.error(f"Error parsing PDF with VLM: {e}")
        import traceback
        traceback.print_exc()
    
    # Save analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"visual_elements_analysis_{timestamp}.json")
    
    report = {
        "pdf_file": str(pdf_path),
        "timestamp": timestamp,
        "without_vlm": {
            "total_segments": document.total_segments if 'document' in locals() else 0,
            "total_visual_elements": document.total_visual_elements if 'document' in locals() else 0,
            "visual_element_types": document.visual_element_types if 'document' in locals() else [],
            "visual_elements": [ve.to_dict() for ve in document.visual_elements] if 'document' in locals() else []
        },
        "with_vlm": {
            "total_segments": document_vlm.total_segments if 'document_vlm' in locals() else 0,
            "total_visual_elements": document_vlm.total_visual_elements if 'document_vlm' in locals() else 0,
            "vlm_analyzed": sum(1 for ve in document_vlm.visual_elements if ve.vlm_description) if 'document_vlm' in locals() else 0,
            "visual_elements": [ve.to_dict() for ve in document_vlm.visual_elements] if 'document_vlm' in locals() else []
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nAnalysis report saved to: {report_path}")


async def main():
    """Main test function"""
    # Use a BMW PDF that likely has images
    test_pdf = Path("/home/bot3/gendocpipe/generic-kg-pipeline/data/input/Preview_BMW_3er_G20.pdf")
    
    if not test_pdf.exists():
        print(f"Test PDF not found: {test_pdf}")
        # Try another PDF
        test_pdf = Path("/home/bot3/gendocpipe/generic-kg-pipeline/data/input/test_simple.pdf")
        if not test_pdf.exists():
            print("No test PDFs found!")
            return
    
    await test_pdf_visual_elements(test_pdf)


if __name__ == "__main__":
    # Activate virtual environment
    import subprocess
    subprocess.run(["source", ".venv/bin/activate"], shell=True)
    
    # Run test
    asyncio.run(main())