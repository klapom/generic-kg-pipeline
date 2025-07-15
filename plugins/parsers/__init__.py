"""Document parser plugins for multi-modal processing"""

from .base_parser import (
    BaseParser, 
    Document, 
    Segment, 
    DocumentMetadata, 
    VisualElement,
    DocumentType,
    VisualElementType,
    ParseError
)

# PDF Parser imports moved to factory to avoid circular dependency
from .docx_parser import DOCXParser
from .xlsx_parser import XLSXParser
from .pptx_parser import PPTXParser
from .parser_factory import ParserFactory, get_default_factory, parse_document, can_parse

__all__ = [
    "BaseParser", 
    "Document", 
    "Segment", 
    "DocumentMetadata", 
    "VisualElement",
    "DocumentType",
    "VisualElementType",
    "ParseError",
    "DOCXParser", 
    "XLSXParser",
    "PPTXParser",
    "ParserFactory",
    "get_default_factory",
    "parse_document",
    "can_parse"
]