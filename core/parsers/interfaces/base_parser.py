"""
Abstract base class for document parsers
"""

import os
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_models import (
    Document, DocumentMetadata, DocumentType, 
    Segment, VisualElement, ParseError
)


class BaseParser(ABC):
    """Abstract base class for multi-modal document parsers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True):
        """
        Initialize parser with optional configuration
        
        Args:
            config: Parser configuration dictionary
            enable_vlm: Whether to enable visual language model analysis
        """
        self.config = config or {}
        self.enable_vlm = enable_vlm
        self._vlm_integration = None
        self.supported_types: set[DocumentType] = set()
        
        # Initialize VLM integration if enabled
        if self.enable_vlm:
            # Lazy import to avoid circular dependency
            from ..vlm_integration import VLMIntegration
            self._vlm_integration = VLMIntegration()
    
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
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if parser can handle this file type
        """
        if not file_path.exists():
            return False
            
        ext = file_path.suffix.lower().lstrip(".")
        for doc_type in self.supported_types:
            if doc_type.value == ext:
                return True
        return False
    
    def analyze_visual_elements(
        self, 
        visual_elements: List[VisualElement],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[VisualElement]:
        """
        Analyze visual elements using VLM integration
        
        Args:
            visual_elements: List of visual elements to analyze
            document_context: Context about the document for better analysis
            
        Returns:
            Updated visual elements with VLM descriptions
        """
        if not self.enable_vlm or not visual_elements or not self._vlm_integration:
            return visual_elements
        
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, so return a coroutine that can be awaited
                # This will be the case when called from hybrid_pdf_parser.parse()
                return self._vlm_integration.analyze_visual_elements(
                    visual_elements,
                    document_context
                )
            except RuntimeError:
                # No running loop, we're in sync context
                # Create new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._vlm_integration.analyze_visual_elements(
                            visual_elements,
                            document_context
                        )
                    )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
        except Exception as e:
            print(f"VLM analysis failed: {e}")
            return visual_elements
    
    def update_visual_segments(
        self,
        segments: List[Segment],
        visual_elements: List[VisualElement]
    ) -> List[Segment]:
        """
        Update visual segments with VLM descriptions
        
        Args:
            segments: List of all segments
            visual_elements: List of analyzed visual elements
            
        Returns:
            Updated segments with visual descriptions
        """
        if not self.enable_vlm or not self._vlm_integration:
            return segments
        
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, so return a coroutine that can be awaited
                # This will be the case when called from hybrid_pdf_parser.parse()
                return self._vlm_integration.update_visual_segments(
                    segments,
                    visual_elements
                )
            except RuntimeError:
                # No running loop, we're in sync context
                # Create new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._vlm_integration.update_visual_segments(
                            segments,
                            visual_elements
                        )
                    )
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
        except Exception as e:
            print(f"Failed to update visual segments: {e}")
            return segments
    
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
        file_path: Path,
        segments: List[Segment],
        metadata: Optional[DocumentMetadata] = None,
        visual_elements: Optional[List[VisualElement]] = None,
        content: Optional[str] = None,
        raw_data: Optional[Any] = None
    ) -> Document:
        """
        Create a Document object with validation
        
        Args:
            file_path: Path to the source file
            segments: List of document segments
            metadata: Document metadata (will be extracted if not provided)
            visual_elements: Optional list of visual elements
            content: Optional full text content
            raw_data: Optional raw document data
            
        Returns:
            Document object
            
        Raises:
            ValueError: If document data is invalid
        """
        if not segments:
            raise ValueError("Document must have at least one segment")
        
        # Extract metadata if not provided
        if metadata is None:
            metadata = self.extract_metadata(file_path)
        
        # Generate document ID
        document_id = f"{metadata.document_type.value}_{file_path.stem}_{hash(file_path)}"
        
        # Build content from segments if not provided
        if content is None:
            content = "\n".join(seg.content for seg in segments)
        
        return Document(
            document_id=document_id,
            source_path=str(file_path),
            document_type=metadata.document_type,
            metadata=metadata,
            segments=segments,
            visual_elements=visual_elements or [],
            content=content,
            raw_data=raw_data
        )
    
    def cleanup(self):
        """Cleanup resources including VLM integration"""
        if self._vlm_integration:
            try:
                self._vlm_integration.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")