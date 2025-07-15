"""
Hybrid PDF Parser compatibility layer

DEPRECATED: This module is deprecated. Import from new location:
- from core.parsers.implementations.pdf import HybridPDFParser
"""

import logging

# Compatibility import
from core.parsers.implementations.pdf import HybridPDFParser

logger = logging.getLogger(__name__)
logger.warning("core.parsers.hybrid_pdf_parser is deprecated. Import from core.parsers.implementations.pdf instead.")

# Re-export for backward compatibility
__all__ = ['HybridPDFParser']