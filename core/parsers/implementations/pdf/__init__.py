"""
PDF parsing implementations

- HybridPDFParser: Complete parser with SmolDocling + fallback + VLM support (RECOMMENDED)
- PDFParser: Multi-modal parser with VLM support (legacy, use HybridPDFParser instead)
"""

from .standard_pdf_parser import PDFParser
from .hybrid_pdf_parser import HybridPDFParser

__all__ = [
    'HybridPDFParser',
    'PDFParser'
]