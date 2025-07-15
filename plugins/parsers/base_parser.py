"""
Compatibility layer for old base_parser imports
This file will be replaced with base_parser.py after migration
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from plugins.parsers.base_parser is deprecated. "
    "Please use core.parsers.interfaces instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new location
from core.parsers.interfaces import (
    BaseParser,
    Document,
    DocumentMetadata,
    DocumentType,
    Segment,
    VisualElement,
    VisualElementType,
    ParseError
)

__all__ = [
    'BaseParser',
    'Document',
    'DocumentMetadata',
    'DocumentType',
    'Segment',
    'VisualElement',
    'VisualElementType',
    'ParseError',
]