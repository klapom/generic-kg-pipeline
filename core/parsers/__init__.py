"""
Core parser module with unified parser hierarchy
"""

# Re-export interfaces for convenience
from .interfaces import (
    BaseParser,
    Document,
    DocumentMetadata,
    DocumentType,
    Segment,
    SegmentType,
    TextSubtype,
    VisualSubtype,
    TableSubtype,
    MetadataSubtype,
    VisualElement,
    VisualElementType,
    ParseError,
    ParserProtocol,
    VisualAnalysisResult
)

# Parser implementations
from .implementations.pdf import PDFParser, HybridPDFParser
from .implementations.office import DOCXParser, XLSXParser, PPTXParser
from .implementations.text import TXTParser

# Factory
from .parser_factory import ParserFactory, get_default_factory, parse_document, can_parse

__all__ = [
    # Interfaces and models
    'BaseParser',
    'Document',
    'DocumentMetadata',
    'DocumentType',
    'Segment',
    'SegmentType',
    'TextSubtype',
    'VisualSubtype',
    'TableSubtype',
    'MetadataSubtype',
    'VisualElement', 
    'VisualElementType',
    'ParseError',
    'ParserProtocol',
    'VisualAnalysisResult',
    
    # Parser implementations
    'PDFParser',
    'HybridPDFParser',
    'DOCXParser',
    'XLSXParser',
    'PPTXParser',
    'TXTParser',
    
    # Factory
    'ParserFactory',
    'get_default_factory',
    'parse_document',
    'can_parse'
]