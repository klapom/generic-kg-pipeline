"""
Data models for document parsing and processing
"""

import hashlib
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


class SegmentType(str, Enum):
    """Main segment types"""
    TEXT = "text"
    VISUAL = "visual"
    TABLE = "table"
    METADATA = "metadata"


class TextSubtype(str, Enum):
    """Subtypes for text segments"""
    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    TITLE = "title"
    SUBTITLE = "subtitle"
    CAPTION = "caption"
    QUOTE = "quote"
    LIST = "list"
    FOOTNOTE = "footnote"
    CODE = "code"
    LINK = "link"


class VisualSubtype(str, Enum):
    """Subtypes for visual segments"""
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    FORMULA = "formula"
    SCREENSHOT = "screenshot"


class TableSubtype(str, Enum):
    """Subtypes for table segments"""
    DATA = "data"
    HEADER = "header"


class MetadataSubtype(str, Enum):
    """Subtypes for metadata segments"""
    DOCUMENT = "document"
    SHEET = "sheet"
    SLIDE = "slide"
    PAGE = "page"


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
    FIGURE = "figure"
    FORMULA = "formula"
    TABLE = "table"
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
    segment_type: Union[SegmentType, str] = SegmentType.TEXT  # Main type
    segment_subtype: Optional[str] = None  # Subtype (flexible string for now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    visual_references: List[str] = field(default_factory=list)  # References to VisualElements
    
    # Legacy field for backwards compatibility
    _legacy_segment_type: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate segment data and handle legacy conversion"""
        if not self.content:
            raise ValueError("Segment content cannot be empty")
        
        # Handle legacy string segment_type
        if isinstance(self.segment_type, str):
            # Only convert if we don't already have a subtype
            if self.segment_subtype is None:
                self._legacy_segment_type = self.segment_type
                self.segment_type, self.segment_subtype = self._convert_legacy_type(self.segment_type)
            else:
                # Convert string to enum if possible
                try:
                    self.segment_type = SegmentType(self.segment_type)
                except ValueError:
                    # If not a valid enum value, treat as legacy
                    self._legacy_segment_type = self.segment_type
                    self.segment_type, self.segment_subtype = self._convert_legacy_type(self.segment_type)
    
    def _convert_legacy_type(self, legacy_type: str) -> tuple[SegmentType, Optional[str]]:
        """Convert legacy segment type to new type/subtype structure"""
        # Mapping for common legacy types
        legacy_mapping = {
            # Text types
            "text": (SegmentType.TEXT, TextSubtype.PARAGRAPH),
            "paragraph": (SegmentType.TEXT, TextSubtype.PARAGRAPH),
            "heading": (SegmentType.TEXT, TextSubtype.HEADING_1),
            "heading_1": (SegmentType.TEXT, TextSubtype.HEADING_1),
            "heading_2": (SegmentType.TEXT, TextSubtype.HEADING_2),
            "heading_3": (SegmentType.TEXT, TextSubtype.HEADING_3),
            "header_1": (SegmentType.TEXT, TextSubtype.HEADING_1),
            "header_2": (SegmentType.TEXT, TextSubtype.HEADING_2),
            "header_3": (SegmentType.TEXT, TextSubtype.HEADING_3),
            "title": (SegmentType.TEXT, TextSubtype.TITLE),
            "subtitle": (SegmentType.TEXT, TextSubtype.SUBTITLE),
            "caption": (SegmentType.TEXT, TextSubtype.CAPTION),
            "quote": (SegmentType.TEXT, TextSubtype.QUOTE),
            "list": (SegmentType.TEXT, TextSubtype.LIST),
            "footnote": (SegmentType.TEXT, TextSubtype.FOOTNOTE),
            "code": (SegmentType.TEXT, TextSubtype.CODE),
            # Table types
            "table": (SegmentType.TABLE, TableSubtype.DATA),
            "data_row": (SegmentType.TABLE, TableSubtype.DATA),
            "header_row": (SegmentType.TABLE, TableSubtype.HEADER),
            "data_range": (SegmentType.TABLE, TableSubtype.DATA),
            # Metadata types
            "sheet_header": (SegmentType.METADATA, MetadataSubtype.SHEET),
            "slide": (SegmentType.METADATA, MetadataSubtype.SLIDE),
            "slide_notes": (SegmentType.TEXT, TextSubtype.FOOTNOTE),
            # Visual types
            "image": (SegmentType.VISUAL, VisualSubtype.IMAGE),
            "chart": (SegmentType.VISUAL, VisualSubtype.CHART),
            "diagram": (SegmentType.VISUAL, VisualSubtype.DIAGRAM),
            "formula": (SegmentType.VISUAL, VisualSubtype.FORMULA),
        }
        
        if legacy_type in legacy_mapping:
            seg_type, subtype = legacy_mapping[legacy_type]
            return seg_type, subtype.value if isinstance(subtype, Enum) else subtype
        
        # Default fallback
        return SegmentType.TEXT, legacy_type


@dataclass
class Document:
    """Represents a parsed multi-modal document"""
    document_id: str  # Unique identifier
    source_path: str  # Original file path
    document_type: DocumentType
    metadata: DocumentMetadata
    segments: List[Segment]
    visual_elements: List[VisualElement] = field(default_factory=list)
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: str = ""  # Full text content (for backward compatibility)
    raw_data: Optional[Any] = None  # Original document data
    
    def __post_init__(self):
        """Initialize derived fields"""
        # If content is empty, build it from segments
        if not self.content and self.segments:
            self.content = "\n".join(seg.content for seg in self.segments)
    
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
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "total_segments": self.total_segments,
            "total_visual_elements": self.total_visual_elements,
            "segment_types": self.segment_types,
            "visual_element_types": self.visual_element_types,
            "pages_with_visuals": len(set(ve.page_or_slide for ve in self.visual_elements if ve.page_or_slide)),
            "vlm_analyzed_elements": sum(1 for ve in self.visual_elements if ve.vlm_description),
            "average_visual_confidence": sum(ve.confidence for ve in self.visual_elements) / len(self.visual_elements) if self.visual_elements else 0.0
        }


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis from VLM models"""
    description: str
    confidence: float
    extracted_data: Optional[Dict[str, Any]] = None
    element_type_detected: Optional[VisualElementType] = None
    ocr_text: Optional[str] = None
    metadata: Dict[str, Any] = None
    processing_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParseError(Exception):
    """Exception raised during document parsing"""
    pass