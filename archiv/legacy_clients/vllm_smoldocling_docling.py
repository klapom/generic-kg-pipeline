"""
VLLMSmolDoclingClient with docling_core integration
Maintains backward compatibility while using official IBM parser
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import logging
import re

from core.clients.vllm_smoldocling_local import (
    VLLMSmolDoclingClient as LegacyVLLMSmolDoclingClient,
    SmolDoclingPage,
    SmolDoclingResult
)

logger = logging.getLogger(__name__)


class VLLMSmolDoclingDoclingClient(LegacyVLLMSmolDoclingClient):
    """
    Enhanced SmolDocling client using docling_core for parsing
    Maintains full backward compatibility with existing pipeline
    """
    
    def __init__(self, *args, use_docling: bool = True, **kwargs):
        """
        Initialize with optional docling support
        
        Args:
            use_docling: Whether to use docling_core for parsing (default: True)
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.use_docling = use_docling
        self._docling_available = self._check_docling_available()
        
        if self.use_docling and not self._docling_available:
            logger.warning("docling_core not available, falling back to legacy parser")
            self.use_docling = False
    
    def _check_docling_available(self) -> bool:
        """Check if docling libraries are available"""
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument
            return True
        except ImportError:
            return False
    
    def parse_model_output(self, output: Any, page_image: Any = None) -> Dict[str, Any]:
        """
        Parse SmolDocling output, optionally using docling_core
        
        Args:
            output: Raw vLLM output
            page_image: PIL.Image for the page (required for docling)
            
        Returns:
            Dict with same structure as legacy parser
        """
        # Use docling if available and image provided
        if self.use_docling and self._docling_available and page_image is not None:
            try:
                return self._parse_with_docling(output, page_image)
            except Exception as e:
                logger.warning(f"Docling parsing failed, using legacy parser: {e}")
                # Fall back to legacy
        
        # Use legacy parser
        return super().parse_model_output(output)
    
    def _parse_with_docling(self, output: Any, page_image: Any) -> Dict[str, Any]:
        """Parse using docling_core libraries"""
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        
        # Extract DocTags text from vLLM output
        doctags_text = self._extract_doctags_text(output)
        
        # Parse with docling
        try:
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                [doctags_text], 
                [page_image]
            )
            
            # Convert to DoclingDocument
            doc = DoclingDocument(name="Page")
            doc.load_from_doctags(doctags_doc)
            
            # Convert to legacy format maintaining bbox
            return self._convert_docling_to_legacy(doc, doctags_text)
            
        except Exception as e:
            logger.error(f"Failed to parse with docling: {e}")
            raise
    
    def _extract_doctags_text(self, output: Any) -> str:
        """Extract raw DocTags text from vLLM output"""
        if hasattr(output, 'outputs') and output.outputs:
            return output.outputs[0].text
        elif hasattr(output, 'text'):
            return output.text
        else:
            return str(output)
    
    def _convert_docling_to_legacy(self, doc: Any, raw_doctags: str) -> Dict[str, Any]:
        """
        Convert DoclingDocument to legacy format
        CRITICAL: Maintains bbox information for PDF image extraction
        """
        result = {
            "text": "",
            "tables": [],
            "images": [],
            "formulas": [],
            "code_blocks": [],
            "text_blocks": [],
            "titles": [],
            "layout_info": {
                "raw_content": raw_doctags,
                "doctags_elements": {},
                "parsed_by": "docling_core"
            },
            "confidence_score": 1.0
        }
        
        all_text = []
        
        # Process DoclingDocument elements
        for element in doc.elements:
            element_type = getattr(element, 'type', 'unknown')
            
            if element_type == 'text':
                text_content = self._get_element_text(element)
                if text_content:
                    # Add to text blocks with bbox if available
                    text_block = {"text": text_content}
                    bbox = self._extract_element_bbox(element)
                    if bbox:
                        text_block["bbox"] = bbox
                    result["text_blocks"].append(text_block)
                    all_text.append(text_content)
            
            elif element_type == 'picture' or element_type == 'image':
                # CRITICAL: Extract and preserve bbox for image extraction
                image_data = {
                    "content": str(element),  # Raw DocTags representation
                    "caption": self._get_element_caption(element),
                }
                
                # Try to get bbox from element
                bbox = self._extract_element_bbox(element)
                if not bbox:
                    # Fallback: Parse from raw DocTags
                    bbox = self._parse_bbox_from_doctags(str(element))
                
                if bbox:
                    image_data["bbox"] = bbox
                    logger.debug(f"Extracted image bbox: {bbox}")
                
                result["images"].append(image_data)
            
            elif element_type == 'table':
                table_text = self._convert_table_to_text(element)
                result["tables"].append({
                    "content": table_text,
                    "format": "text"
                })
                all_text.append(table_text)
            
            elif element_type == 'formula':
                formula_data = {
                    "latex": self._get_element_text(element),
                    "type": "formula"
                }
                result["formulas"].append(formula_data)
            
            elif element_type in ['title', 'heading', 'section_header']:
                title_text = self._get_element_text(element)
                if title_text:
                    result["titles"].append(title_text)
                    all_text.append(title_text)
        
        # Build full text
        result["text"] = "\n\n".join(all_text)
        
        # Count elements
        result["layout_info"]["doctags_elements"] = {
            "text_count": len(result["text_blocks"]),
            "table_count": len(result["tables"]),
            "picture_count": len(result["images"]),
            "formula_count": len(result["formulas"]),
            "code_count": len(result["code_blocks"])
        }
        
        return result
    
    def _extract_element_bbox(self, element: Any) -> Optional[List[int]]:
        """Extract bbox from DoclingDocument element"""
        # Check for bbox attribute
        if hasattr(element, 'bbox'):
            bbox = element.bbox
            if hasattr(bbox, 'x0'):
                # Structured bbox object
                return [int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)]
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                # Already a list
                return [int(x) for x in bbox[:4]]
        
        # Check for location attributes
        if hasattr(element, 'location'):
            loc = element.location
            if hasattr(loc, 'x'):
                return [int(loc.x), int(loc.y), 
                       int(loc.x + getattr(loc, 'width', 0)), 
                       int(loc.y + getattr(loc, 'height', 0))]
        
        return None
    
    def _parse_bbox_from_doctags(self, doctags: str) -> Optional[List[int]]:
        """Parse bbox from raw DocTags string"""
        # Look for <loc_x><loc_y><loc_x2><loc_y2> pattern
        loc_match = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', doctags)
        if loc_match:
            return [int(loc_match.group(i)) for i in range(1, 5)]
        
        # Look for bare coordinate pattern x>y>x2>y2>
        coord_match = re.search(r'(\d+)>(\d+)>(\d+)>(\d+)>', doctags)
        if coord_match:
            return [int(coord_match.group(i)) for i in range(1, 5)]
        
        return None
    
    def _get_element_text(self, element: Any) -> str:
        """Extract text content from element"""
        if hasattr(element, 'text'):
            return element.text
        elif hasattr(element, 'content'):
            return element.content
        elif hasattr(element, 'value'):
            return element.value
        else:
            return str(element)
    
    def _get_element_caption(self, element: Any) -> str:
        """Extract caption from element"""
        if hasattr(element, 'caption'):
            return element.caption or ""
        return ""
    
    def _convert_table_to_text(self, element: Any) -> str:
        """Convert table element to text representation"""
        # Try different methods
        if hasattr(element, 'to_markdown'):
            return element.to_markdown()
        elif hasattr(element, 'to_text'):
            return element.to_text()
        elif hasattr(element, 'export_to_markdown'):
            return element.export_to_markdown()
        else:
            # Fallback to string representation
            return str(element)
    
    def parse_pdf(self, pdf_path: Path) -> SmolDoclingResult:
        """
        Parse PDF with optional docling support
        
        Overrides parent to pass page images to parse_model_output
        """
        if not self.use_docling or not self._docling_available:
            # Use legacy implementation
            return super().parse_pdf(pdf_path)
        
        # Custom implementation that passes images to parser
        # This would require modifying the generation loop to keep page images
        # For now, fall back to legacy
        logger.info("Using legacy parser for full PDF (docling requires page image passing)")
        return super().parse_pdf(pdf_path)


# Convenience function for testing
def create_docling_client(**kwargs) -> VLLMSmolDoclingDoclingClient:
    """Create a SmolDocling client with docling support"""
    return VLLMSmolDoclingDoclingClient(
        model_id="smoldocling",
        use_docling=kwargs.pop('use_docling', True),
        **kwargs
    )