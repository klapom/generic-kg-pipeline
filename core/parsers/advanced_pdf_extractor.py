"""
Advanced PDF extraction with bounding box filtering and layout preservation

DEPRECATED: This module is deprecated. The functionality has been integrated into:
- from core.parsers.implementations.pdf.extractors.pdfplumber_extractor import PDFPlumberExtractor

Use PDFPlumberExtractor with use_bbox_filtering=True for advanced extraction.
"""

import logging

# Compatibility alias
from core.parsers.implementations.pdf.extractors.pdfplumber_extractor import PDFPlumberExtractor

# Create alias for backward compatibility
AdvancedPDFExtractor = PDFPlumberExtractor

logger = logging.getLogger(__name__)
logger.warning("core.parsers.advanced_pdf_extractor is deprecated. Use PDFPlumberExtractor with use_bbox_filtering=True instead.")

# Re-export for backward compatibility
__all__ = ['AdvancedPDFExtractor']