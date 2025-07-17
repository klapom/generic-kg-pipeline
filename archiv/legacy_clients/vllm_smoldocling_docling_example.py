"""
Example implementation of SmolDocling parser using official docling libraries
This is a proof of concept for the migration
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import logging

# Future imports after adding to requirements.txt
# from docling_core.types.doc import DoclingDocument
# from docling_core.types.doc.document import DocTagsDocument
# import PIL.Image

from core.clients.vllm_smoldocling_local import (
    SmolDoclingPage as LegacySmolDoclingPage,
    SmolDoclingResult as LegacySmolDoclingResult
)

logger = logging.getLogger(__name__)


@dataclass
class DoclingSmolDoclingPage:
    """
    SmolDocling page with docling backend
    Maintains backward compatibility with existing API
    """
    page_number: int
    raw_doctags: str  # Raw DocTags XML output
    image: Any  # PIL.Image in real implementation
    _docling_doc: Optional[Any] = None  # DoclingDocument when available
    _legacy_data: Optional[Dict[str, Any]] = None  # Fallback data
    
    @property
    def text(self) -> str:
        """Get page text - backward compatible"""
        if self._docling_doc:
            # Use docling export
            # return self._docling_doc.export_to_markdown()
            pass
        return self._legacy_data.get("text", "") if self._legacy_data else ""
    
    @property
    def tables(self) -> List[Dict[str, Any]]:
        """Get tables - backward compatible"""
        if self._docling_doc:
            # Extract from DoclingDocument
            # return self._extract_tables_from_docling()
            pass
        return self._legacy_data.get("tables", []) if self._legacy_data else []
    
    @property
    def images(self) -> List[Dict[str, Any]]:
        """Get images with bounding boxes"""
        if self._docling_doc:
            # Extract from DoclingDocument with proper bbox
            # return self._extract_images_from_docling()
            pass
        return self._legacy_data.get("images", []) if self._legacy_data else []
    
    def to_markdown(self) -> str:
        """Export page as markdown"""
        if self._docling_doc:
            # return self._docling_doc.export_to_markdown()
            pass
        return self.text
    
    def to_json(self) -> dict:
        """Export page as structured JSON"""
        if self._docling_doc:
            # return self._docling_doc.export_to_json()
            pass
        return {
            "page_number": self.page_number,
            "text": self.text,
            "tables": self.tables,
            "images": self.images
        }


@dataclass
class DoclingSmolDoclingResult:
    """
    Complete parsing result with docling backend
    Maintains backward compatibility
    """
    pages: List[DoclingSmolDoclingPage]
    metadata: Dict[str, Any]
    processing_time_seconds: float
    model_version: str
    total_pages: int
    success: bool
    error_message: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Export entire document as markdown"""
        return "\n\n---\n\n".join(
            page.to_markdown() for page in self.pages
        )
    
    def to_legacy_format(self) -> LegacySmolDoclingResult:
        """Convert to legacy format for backward compatibility"""
        legacy_pages = []
        for page in self.pages:
            legacy_page = LegacySmolDoclingPage(
                page_number=page.page_number,
                text=page.text,
                tables=page.tables,
                images=page.images,
                formulas=page._legacy_data.get("formulas", []) if page._legacy_data else [],
                layout_info=page._legacy_data.get("layout_info", {}) if page._legacy_data else {},
                confidence_score=page._legacy_data.get("confidence_score", 1.0) if page._legacy_data else 1.0
            )
            legacy_pages.append(legacy_page)
        
        return LegacySmolDoclingResult(
            pages=legacy_pages,
            metadata=self.metadata,
            processing_time_seconds=self.processing_time_seconds,
            model_version=self.model_version,
            total_pages=self.total_pages,
            success=self.success,
            error_message=self.error_message
        )


class VLLMSmolDoclingDoclingClient:
    """
    SmolDocling client using official docling libraries
    Drop-in replacement for VLLMSmolDoclingClient
    """
    
    def __init__(self, *args, **kwargs):
        # Import base class dynamically to avoid circular imports
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        self._legacy_client = VLLMSmolDoclingClient(*args, **kwargs)
        self.use_docling = kwargs.get("use_docling", True)
        
    def parse_with_docling(
        self, 
        doctags: str, 
        image: Any,  # PIL.Image
        page_number: int
    ) -> DoclingSmolDoclingPage:
        """
        Parse single page using docling libraries
        
        This is where the magic happens:
        1. Use DocTagsDocument.from_doctags_and_image_pairs
        2. Convert to DoclingDocument
        3. Wrap in our compatibility layer
        """
        try:
            # When docling is available:
            # from docling_core.types.doc.document import DocTagsDocument
            # from docling_core.types.doc import DoclingDocument
            
            # # Parse DocTags
            # doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            #     [doctags], 
            #     [image]
            # )
            # 
            # # Convert to DoclingDocument
            # doc = DoclingDocument(name=f"Page_{page_number}")
            # doc.load_from_doctags(doctags_doc)
            # 
            # # Create our page wrapper
            # page = DoclingSmolDoclingPage(
            #     page_number=page_number,
            #     raw_doctags=doctags,
            #     image=image,
            #     _docling_doc=doc
            # )
            
            # For now, fall back to legacy parsing
            raise NotImplementedError("Docling not yet installed")
            
        except Exception as e:
            logger.warning(f"Docling parsing failed, using legacy parser: {e}")
            # Fall back to legacy parsing
            legacy_output = self._legacy_client.parse_model_output(doctags)
            
            page = DoclingSmolDoclingPage(
                page_number=page_number,
                raw_doctags=doctags,
                image=image,
                _legacy_data=legacy_output
            )
            
        return page
    
    def parse_pdf(self, pdf_path: Path) -> DoclingSmolDoclingResult:
        """
        Parse PDF with docling backend
        Maintains same API as legacy client
        """
        start_time = time.time()
        
        if not self.use_docling:
            # Use legacy implementation directly
            legacy_result = self._legacy_client.parse_pdf(pdf_path)
            # Convert to new format (would need proper conversion)
            return self._convert_legacy_result(legacy_result)
        
        try:
            # This would use the same vLLM generation as before
            # but parse output with docling
            
            # For demonstration, we'll use legacy client for generation
            # but show how parsing would change
            
            # Get raw DocTags output from vLLM (reuse existing code)
            # ... vLLM generation code ...
            
            # Parse each page with docling
            pages = []
            # for page_num, (doctags, image) in enumerate(results):
            #     page = self.parse_with_docling(doctags, image, page_num + 1)
            #     pages.append(page)
            
            # For now, use legacy and convert
            legacy_result = self._legacy_client.parse_pdf(pdf_path)
            return self._convert_legacy_result(legacy_result)
            
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            return DoclingSmolDoclingResult(
                pages=[],
                metadata={"error": str(e)},
                processing_time_seconds=time.time() - start_time,
                model_version="docling-example",
                total_pages=0,
                success=False,
                error_message=str(e)
            )
    
    def _convert_legacy_result(
        self, 
        legacy_result: LegacySmolDoclingResult
    ) -> DoclingSmolDoclingResult:
        """Convert legacy result to new format"""
        pages = []
        for legacy_page in legacy_result.pages:
            page = DoclingSmolDoclingPage(
                page_number=legacy_page.page_number,
                raw_doctags="",  # Not available in legacy
                image=None,  # Not stored in legacy
                _legacy_data={
                    "text": legacy_page.text,
                    "tables": legacy_page.tables,
                    "images": legacy_page.images,
                    "formulas": legacy_page.formulas,
                    "layout_info": legacy_page.layout_info,
                    "confidence_score": legacy_page.confidence_score
                }
            )
            pages.append(page)
        
        return DoclingSmolDoclingResult(
            pages=pages,
            metadata=legacy_result.metadata,
            processing_time_seconds=legacy_result.processing_time_seconds,
            model_version=legacy_result.model_version,
            total_pages=legacy_result.total_pages,
            success=legacy_result.success,
            error_message=legacy_result.error_message
        )


# Example usage when docling is available:
"""
def example_with_real_docling():
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    from pdf2image import convert_from_path
    
    # Get SmolDocling output (DocTags XML)
    doctags = '<doctag><picture><loc_0><loc_0><loc_500><loc_375>...</doctag>'
    
    # Get corresponding image
    images = convert_from_path("document.pdf", dpi=144)
    image = images[0]
    
    # Parse with docling
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        [doctags], 
        [image]
    )
    
    # Convert to DoclingDocument
    doc = DoclingDocument(name="MyDocument")
    doc.load_from_doctags(doctags_doc)
    
    # Export in various formats
    markdown = doc.export_to_markdown()
    html = doc.export_to_html()
    json_data = doc.export_to_json()
    
    # Access structured elements
    for element in doc.elements:
        if element.type == "table":
            # Process table
            pass
        elif element.type == "image":
            # Process image with bbox
            pass
"""