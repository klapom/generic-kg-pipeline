#!/usr/bin/env python3
"""
Test docling with explicit image extraction settings
"""

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_docling_images():
    """Test docling with different settings for image extraction"""
    
    # Find a BMW PDF
    pdf_path = None
    for pattern in ["*BMW*.pdf", "*bmw*.pdf"]:
        files = list(Path("data/input").glob(pattern))
        if files:
            pdf_path = files[0]
            break
    
    if not pdf_path:
        logger.error("No BMW PDF found")
        return
    
    logger.info(f"Testing with: {pdf_path}")
    
    # Try with explicit pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Disable OCR for speed
    pipeline_options.images_scale = 2.0  # Higher quality images
    pipeline_options.generate_page_images = True  # Generate page images
    pipeline_options.generate_picture_images = True  # Generate picture images
    
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        pipeline_options={InputFormat.PDF: pipeline_options}
    )
    
    logger.info("Converting with image extraction enabled...")
    result = converter.convert(str(pdf_path))
    
    logger.info(f"\nConversion complete:")
    logger.info(f"  Document type: {type(result)}")
    
    # Check main document structure
    if hasattr(result, 'pages'):
        logger.info(f"  Pages: {len(result.pages)}")
        
        # Check page-level content
        for i, page in enumerate(result.pages[:3]):
            logger.info(f"\nPage {i+1}:")
            logger.info(f"  Page size: {page.size if hasattr(page, 'size') else 'unknown'}")
            
            # Check for different types of content
            if hasattr(page, 'children'):
                logger.info(f"  Children: {len(page.children)}")
                
                # Count different types
                type_counts = {}
                for child in page.children:
                    child_type = type(child).__name__
                    type_counts[child_type] = type_counts.get(child_type, 0) + 1
                
                for child_type, count in type_counts.items():
                    logger.info(f"    {child_type}: {count}")
    
    # Check for pictures in the main document
    if hasattr(result, 'pictures'):
        logger.info(f"\nDocument pictures: {len(result.pictures)}")
        for i, pic in enumerate(result.pictures[:3]):
            logger.info(f"  Picture {i+1}: {pic}")
    
    # Export to DocTags to see the structure
    logger.info("\nExporting to DocTags...")
    doctags = result.export_to_doctags()
    
    # Count picture tags
    picture_count = doctags.count('<picture>')
    logger.info(f"Picture tags in DocTags: {picture_count}")
    
    # Find first picture tag if any
    if picture_count > 0:
        start = doctags.find('<picture>')
        end = doctags.find('</picture>', start) + len('</picture>')
        logger.info(f"\nFirst picture tag:")
        logger.info(doctags[start:end])
    
    # Also check the document structure
    logger.info("\nDocument structure:")
    for attr in dir(result):
        if not attr.startswith('_') and 'picture' in attr.lower():
            logger.info(f"  Found attribute: {attr}")
            value = getattr(result, attr)
            if hasattr(value, '__len__'):
                logger.info(f"    Length: {len(value)}")


if __name__ == "__main__":
    test_docling_images()