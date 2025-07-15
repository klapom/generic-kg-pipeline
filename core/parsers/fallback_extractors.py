"""
Fallback text extractors for complex PDF layouts

DEPRECATED: This module is deprecated. Import from new location:
- from core.parsers.implementations.pdf.extractors.pdfplumber_extractor import PDFPlumberExtractor

Note: PyPDF2TextExtractor has been removed. Use PDFPlumberExtractor instead.
"""

import logging

# Compatibility import
from core.parsers.implementations.pdf.extractors.pdfplumber_extractor import PDFPlumberExtractor

logger = logging.getLogger(__name__)
logger.warning("core.parsers.fallback_extractors is deprecated. Import from core.parsers.implementations.pdf.extractors instead.")

# PyPDF2TextExtractor removed - use PDFPlumberExtractor instead
PyPDF2TextExtractor = PDFPlumberExtractor  # Alias for backward compatibility

# Re-export for backward compatibility
__all__ = ['PyPDF2TextExtractor', 'PDFPlumberExtractor']