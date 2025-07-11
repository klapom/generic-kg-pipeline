"""Base parser interface and data structures for multi-modal document processing"""

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"
    UNKNOWN = "unknown"


class VisualElementType(Enum):
    """Types of visual elements that can be analyzed"""
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    GRAPH = "graph"
    TABLE_IMAGE = "table_image"
    SCREENSHOT = "screenshot"
    DRAWING = "drawing"
    MAP = "map"
    UNKNOWN_VISUAL = "unknown_visual"


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
class VisualElement:
    """A visual element (image, chart, diagram) extracted from a document"""
    element_type: VisualElementType
    source_format: DocumentType
    content_hash: str  # Hash of visual content for deduplication
    vlm_description: Optional[str] = None  # Qwen2.5-VL generated description
    extracted_data: Optional[Dict[str, Any]] = None  # Structured data if chart/table
    confidence: float = 0.0  # VLM analysis confidence
    bounding_box: Optional[Dict[str, float]] = None  # x, y, width, height
    page_or_slide: Optional[int] = None
    segment_reference: Optional[str] = None  # Reference to related text segment
    file_extension: str = "png"  # Default image format
    raw_data: Optional[bytes] = None  # Actual image bytes
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_hash(cls, content: bytes) -> str:
        """Create content hash for visual element"""
        return hashlib.md5(content).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (excluding raw_data)"""
        return {
            "element_type": self.element_type.value,
            "source_format": self.source_format.value,
            "content_hash": self.content_hash,
            "vlm_description": self.vlm_description,
            "extracted_data": self.extracted_data,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "page_or_slide": self.page_or_slide,
            "segment_reference": self.segment_reference,
            "file_extension": self.file_extension,
            "analysis_metadata": self.analysis_metadata
        }


@dataclass
class Segment:
    """A segment of text from a document"""
    content: str
    page_number: Optional[int] = None
    segment_index: int = 0
    segment_type: str = "text"  # text, table, image_caption, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    visual_references: List[str] = field(default_factory=list)  # References to VisualElements
    
    def __post_init__(self):
        """Validate segment data"""
        if not self.content:
            raise ValueError("Segment content cannot be empty")


@dataclass
class Document:
    """Represents a parsed multi-modal document"""
    content: str
    segments: List[Segment]
    metadata: DocumentMetadata
    visual_elements: List[VisualElement] = field(default_factory=list)  # NEW: Visual content
    raw_data: Optional[Any] = None  # Original document data
    
    @property
    def total_segments(self) -> int:
        """Get total number of segments"""
        return len(self.segments)
    
    @property
    def total_visual_elements(self) -> int:
        """Get total number of visual elements"""
        return len(self.visual_elements)
    
    @property
    def segment_types(self) -> List[str]:
        """Get unique segment types in document"""
        return list(set(seg.segment_type for seg in self.segments))
    
    @property
    def visual_element_types(self) -> List[str]:
        """Get unique visual element types in document"""
        return list(set(ve.element_type.value for ve in self.visual_elements))
    
    def get_segments_by_type(self, segment_type: str) -> List[Segment]:
        """Get all segments of a specific type"""
        return [seg for seg in self.segments if seg.segment_type == segment_type]
    
    def get_segments_by_page(self, page_number: int) -> List[Segment]:
        """Get all segments from a specific page"""
        return [seg for seg in self.segments if seg.page_number == page_number]
    
    def get_visual_elements_by_type(self, element_type: Union[VisualElementType, str]) -> List[VisualElement]:
        """Get all visual elements of a specific type"""
        if isinstance(element_type, str):
            return [ve for ve in self.visual_elements if ve.element_type.value == element_type]
        return [ve for ve in self.visual_elements if ve.element_type == element_type]
    
    def get_visual_elements_by_page(self, page_number: int) -> List[VisualElement]:
        """Get all visual elements from a specific page/slide"""
        return [ve for ve in self.visual_elements if ve.page_or_slide == page_number]
    
    def get_enhanced_content(self) -> str:
        """Get full content including VLM descriptions of visual elements"""
        content_parts = [self.content]
        
        for visual in self.visual_elements:
            if visual.vlm_description:
                visual_content = f"\n[{visual.element_type.value.upper()}]: {visual.vlm_description}"
                if visual.page_or_slide:
                    visual_content += f" (Page/Slide {visual.page_or_slide})"
                content_parts.append(visual_content)
        
        return "\n".join(content_parts)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of document analysis"""
        return {
            "total_segments": self.total_segments,
            "total_visual_elements": self.total_visual_elements,
            "segment_types": self.segment_types,
            "visual_element_types": self.visual_element_types,
            "pages_with_visuals": len(set(ve.page_or_slide for ve in self.visual_elements if ve.page_or_slide)),
            "vlm_analyzed_elements": sum(1 for ve in self.visual_elements if ve.vlm_description),
            "average_visual_confidence": sum(ve.confidence for ve in self.visual_elements) / len(self.visual_elements) if self.visual_elements else 0.0
        }


class ParseError(Exception):
    """Exception raised during document parsing"""
    pass


class BaseParser(ABC):
    """Abstract base class for multi-modal document parsers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True):
        """Initialize parser with optional configuration"""
        self.config = config or {}
        self.enable_vlm = enable_vlm
        self._vlm_client = None
    
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """
        Parse a document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Parsed Document object with visual elements
            
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
    
    async def analyze_visual_elements(
        self, 
        visual_elements: List[VisualElement],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[VisualElement]:
        """
        Analyze visual elements using Qwen2.5-VL
        
        Args:
            visual_elements: List of visual elements to analyze
            document_context: Context about the document for better analysis
            
        Returns:
            Updated visual elements with VLM descriptions
        """
        if not self.enable_vlm or not visual_elements:
            return visual_elements
        
        # Import VLM client here to avoid circular imports
        try:
            from core.clients.qwen25_vl import Qwen25VLClient
            
            async with Qwen25VLClient() as vlm_client:
                analyzed_elements = []
                
                for element in visual_elements:
                    try:
                        if element.raw_data:
                            analysis_result = await vlm_client.analyze_visual(
                                image_data=element.raw_data,
                                document_context=document_context,
                                element_type=element.element_type
                            )
                            
                            # Update element with VLM analysis
                            element.vlm_description = analysis_result.description
                            element.confidence = analysis_result.confidence
                            element.extracted_data = analysis_result.extracted_data
                            element.analysis_metadata = analysis_result.metadata
                            
                        analyzed_elements.append(element)
                        
                    except Exception as e:
                        # Log error but continue with other elements
                        print(f"VLM analysis failed for element {element.content_hash}: {e}")
                        analyzed_elements.append(element)
                
                return analyzed_elements
                
        except ImportError:
            # VLM client not available, return elements unchanged
            print("Qwen2.5-VL client not available, skipping visual analysis")
            return visual_elements
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return visual_elements
    
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
        visual_elements: Optional[List[VisualElement]] = None,
        raw_data: Optional[Any] = None
    ) -> Document:
        """
        Create a Document object with validation
        
        Args:
            content: Full document content
            segments: List of document segments
            metadata: Document metadata
            visual_elements: Optional list of visual elements
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
            visual_elements=visual_elements or [],
            raw_data=raw_data
        )


import os  # Import needed for os.access in validate_file method