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
    VisualElement,
    VisualElementType,
    ParseError,
    ParserProtocol
)

# Note: Parser implementations will be added here after migration
# from .implementations.pdf import PDFParser, HybridPDFParser
# from .implementations.office import DOCXParser, XLSXParser, PPTXParser
# from .implementations.text import TXTParser
# from .factory import get_parser, register_parser

__all__ = [
    # Interfaces and models
    'BaseParser',
    'Document',
    'DocumentMetadata',
    'DocumentType',
    'Segment',
    'VisualElement', 
    'VisualElementType',
    'ParseError',
    'ParserProtocol',
    
    # Implementations (to be added)
    # 'PDFParser',
    # 'HybridPDFParser',
    # 'DOCXParser',
    # 'XLSXParser',
    # 'PPTXParser',
    # 'TXTParser',
    
    # Factory (to be added)
    # 'get_parser',
    # 'register_parser',
]