"""
Parser interfaces and data models
"""

from .data_models import (
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
    VisualAnalysisResult
)

from .base_parser import BaseParser
from .parser_protocol import ParserProtocol

__all__ = [
    # Data models
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
    'VisualAnalysisResult',
    
    # Interfaces
    'BaseParser',
    'ParserProtocol',
]