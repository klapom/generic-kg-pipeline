"""
PDF parser compatibility layer

DEPRECATED: This module provides backward compatibility. 
Use the new parsers from:
- from core.parsers.implementations.pdf import PDFParser
- from core.parsers.implementations.pdf import HybridPDFParser
"""

import logging

# Compatibility imports
from core.parsers.implementations.pdf import PDFParser
from core.parsers.implementations.pdf import HybridPDFParser

logger = logging.getLogger(__name__)
logger.warning("plugins.parsers.pdf_parser is deprecated. Import from core.parsers.implementations.pdf instead.")

# Re-export for backward compatibility
__all__ = ['PDFParser', 'HybridPDFParser']