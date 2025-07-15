"""
Parser Protocol for type checking
"""

from typing import Protocol, runtime_checkable
from pathlib import Path

from .data_models import Document


@runtime_checkable
class ParserProtocol(Protocol):
    """Protocol defining the interface for document parsers"""
    
    def parse(self, file_path: Path) -> Document:
        """Parse a document from file path"""
        ...
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        ...