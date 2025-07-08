"""Base parser interface and data structures"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    file_size: Optional[int] = None
    file_path: Optional[Path] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    """A segment of text from a document"""
    content: str
    page_number: Optional[int] = None
    segment_index: int = 0
    segment_type: str = "text"  # text, table, image_caption, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate segment data"""
        if not self.content:
            raise ValueError("Segment content cannot be empty")


@dataclass
class Document:
    """Represents a parsed document"""
    content: str
    segments: List[Segment]
    metadata: DocumentMetadata
    raw_data: Optional[Any] = None  # Original document data
    
    @property
    def total_segments(self) -> int:
        """Get total number of segments"""
        return len(self.segments)
    
    @property
    def segment_types(self) -> List[str]:
        """Get unique segment types in document"""
        return list(set(seg.segment_type for seg in self.segments))
    
    def get_segments_by_type(self, segment_type: str) -> List[Segment]:
        """Get all segments of a specific type"""
        return [seg for seg in self.segments if seg.segment_type == segment_type]
    
    def get_segments_by_page(self, page_number: int) -> List[Segment]:
        """Get all segments from a specific page"""
        return [seg for seg in self.segments if seg.page_number == page_number]


class ParseError(Exception):
    """Exception raised during document parsing"""
    pass


class BaseParser(ABC):
    """Abstract base class for document parsers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with optional configuration"""
        self.config = config or {}
    
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """
        Parse a document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Parsed Document object
            
        Raises:
            ParseError: If parsing fails
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if parser can handle this file type
        """
        pass
    
    def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Extract basic metadata from file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentMetadata object with basic information
        """
        stat = file_path.stat()
        
        # Determine document type from extension
        ext = file_path.suffix.lower().lstrip(".")
        doc_type = DocumentType.UNKNOWN
        for dtype in DocumentType:
            if dtype.value == ext:
                doc_type = dtype
                break
        
        return DocumentMetadata(
            file_path=file_path,
            file_size=stat.st_size,
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            created_date=datetime.fromtimestamp(stat.st_ctime),
            document_type=doc_type,
            title=file_path.stem
        )
    
    def validate_file(self, file_path: Path) -> None:
        """
        Validate that file exists and is readable
        
        Args:
            file_path: Path to the document file
            
        Raises:
            ParseError: If file validation fails
        """
        if not file_path.exists():
            raise ParseError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ParseError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise ParseError(f"File is not readable: {file_path}")
    
    def create_document(
        self,
        content: str,
        segments: List[Segment],
        metadata: DocumentMetadata,
        raw_data: Optional[Any] = None
    ) -> Document:
        """
        Create a Document object with validation
        
        Args:
            content: Full document content
            segments: List of document segments
            metadata: Document metadata
            raw_data: Optional raw document data
            
        Returns:
            Document object
            
        Raises:
            ValueError: If document data is invalid
        """
        if not content:
            raise ValueError("Document content cannot be empty")
        
        if not segments:
            # Create a single segment if none provided
            segments = [Segment(content=content, segment_index=0)]
        
        return Document(
            content=content,
            segments=segments,
            metadata=metadata,
            raw_data=raw_data
        )


import os  # Import needed for os.access in validate_file method