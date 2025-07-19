#!/usr/bin/env python3
"""
Quick check of SmolDocling visual element extraction
"""

from pathlib import Path
from docling.document_converter import DocumentConverter

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_smoldocling():
    """Check SmolDocling directly without our wrapper"""
    
    # Test with BMW document
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    if not pdf_path.exists():
        # Try other BMW files
        for pattern in ["*BMW*.pdf", "*bmw*.pdf"]:
            files = list(Path("data/input").glob(pattern))
            if files:
                pdf_path = files[0]
                break
    
    logger.info(f"Testing with: {pdf_path}")
    
    # Use SmolDocling directly
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    
    logger.info(f"\nSmolDocling results:")
    logger.info(f"  Document type: {type(result)}")
    
    # Check for visual elements
    if hasattr(result, 'pages'):
        logger.info(f"  Pages: {len(result.pages)}")
        
        for i, page in enumerate(result.pages[:3]):
            logger.info(f"\nPage {i+1}:")
            if hasattr(page, 'images'):
                logger.info(f"  Images: {len(page.images)}")
            if hasattr(page, 'tables'):
                logger.info(f"  Tables: {len(page.tables)}")
            if hasattr(page, 'figures'):
                logger.info(f"  Figures: {len(page.figures)}")
    
    # Check exported content
    content = result.export_to_markdown()
    logger.info(f"\nMarkdown export length: {len(content)}")
    
    # Look for image indicators
    import re
    image_count = len(re.findall(r'!\[.*?\]\(.*?\)', content))
    logger.info(f"Markdown images found: {image_count}")
    
    # Check DocTags format
    doctags = result.export_to_doctags()
    logger.info(f"\nDocTags export length: {len(doctags)}")
    
    # Count visual indicators in DocTags
    image_tags = doctags.count('<image>')
    figure_tags = doctags.count('<figure>')
    logger.info(f"DocTags image tags: {image_tags}")
    logger.info(f"DocTags figure tags: {figure_tags}")


if __name__ == "__main__":
    check_smoldocling()