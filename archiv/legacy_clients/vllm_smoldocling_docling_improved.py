"""
Improved VLLMSmolDoclingClient with direct image extraction from Docling
This approach extracts images directly when we have the DoclingDocument
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging
import re
import io

from core.clients.vllm_smoldocling_local import (
    VLLMSmolDoclingClient as LegacyVLLMSmolDoclingClient,
    SmolDoclingPage,
    SmolDoclingResult
)
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

logger = logging.getLogger(__name__)


class VLLMSmolDoclingDoclingImprovedClient(LegacyVLLMSmolDoclingClient):
    """
    Enhanced SmolDocling client that extracts images directly from Docling
    """
    
    def __init__(self, *args, use_docling: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_docling = use_docling
        self._docling_available = self._check_docling_available()
        
    def _check_docling_available(self) -> bool:
        """Check if docling libraries are available"""
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument
            return True
        except ImportError:
            return False
    
    def parse_pdf_with_docling(self, pdf_path: Path) -> SmolDoclingResult:
        """
        Parse PDF using Docling with integrated image extraction
        This is the improved approach!
        """
        if not self.use_docling or not self._docling_available:
            return super().parse_pdf(pdf_path)
        
        start_time = time.time()
        pages = []
        all_visual_elements = []
        
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument
            from pdf2image import convert_from_path
            import fitz  # PyMuPDF for image extraction
            
            # Convert PDF to images
            logger.info("Converting PDF to images...")
            page_images = convert_from_path(
                str(pdf_path),
                dpi=144,  # SmolDocling optimal DPI
                fmt='PNG'
            )
            
            # Open PDF with PyMuPDF for image extraction
            pdf_doc = fitz.open(str(pdf_path))
            
            # Process each page
            for page_num, page_image in enumerate(page_images, 1):
                logger.info(f"Processing page {page_num}...")
                
                # Generate DocTags with vLLM
                doctags = self._generate_doctags_for_page(page_image, page_num)
                
                # Parse with Docling
                doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                    [doctags], 
                    [page_image]
                )
                
                # Convert to DoclingDocument
                doc = DoclingDocument(name=f"Page_{page_num}")
                doc.load_from_doctags(doctags_doc)
                
                # Extract visual elements WITH images directly
                visual_elements = self._extract_visuals_from_docling(
                    doc, 
                    pdf_doc,
                    page_num,
                    page_image
                )
                all_visual_elements.extend(visual_elements)
                
                # Create page data
                page_data = self._create_page_from_docling(
                    doc,
                    page_num,
                    visual_elements
                )
                pages.append(page_data)
            
            pdf_doc.close()
            
            # Create result
            return SmolDoclingResult(
                pages=pages,
                metadata={
                    "filename": pdf_path.name,
                    "total_visual_elements": len(all_visual_elements)
                },
                processing_time_seconds=time.time() - start_time,
                model_version="docling-improved",
                total_pages=len(pages),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            # Fallback to legacy
            return super().parse_pdf(pdf_path)
    
    def _generate_doctags_for_page(self, page_image: Any, page_num: int) -> str:
        """Generate DocTags for a single page using vLLM"""
        # This would use the existing vLLM generation logic
        # For now, simplified
        multimodal_input = {
            "prompt": f"<|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:",
            "multi_modal_data": {"image": page_image}
        }
        
        # Generate with vLLM (simplified)
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False
        )
        
        # This would use the actual model
        # outputs = self.model.generate(multimodal_input, sampling_params)
        # return outputs[0].outputs[0].text
        
        # Mock for now
        return "<doctag>...</doctag>"
    
    def _extract_visuals_from_docling(
        self, 
        doc: Any,  # DoclingDocument
        pdf_doc: Any,  # fitz.Document
        page_num: int,
        page_image: Any  # PIL.Image
    ) -> List[VisualElement]:
        """
        Extract visual elements WITH image bytes directly from Docling
        This is the KEY IMPROVEMENT!
        """
        visual_elements = []
        page = pdf_doc[page_num - 1]
        
        # Get page dimensions for scaling
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        scale_x = page_width / 500.0
        scale_y = page_height / 500.0
        
        # Iterate through Docling elements
        for element in doc.elements:
            if element.type in ['picture', 'image', 'figure']:
                # Extract bbox from element
                bbox = self._get_element_bbox(element)
                
                if bbox:
                    # Scale bbox to page coordinates
                    x0 = bbox[0] * scale_x
                    y0 = bbox[1] * scale_y
                    x1 = bbox[2] * scale_x
                    y1 = bbox[3] * scale_y
                    
                    # Extract image directly!
                    rect = fitz.Rect(x0, y0, x1, y1)
                    mat = fitz.Matrix(2, 2)  # 2x zoom
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    img_bytes = pix.tobytes("png")
                    
                    # Create VisualElement with image data
                    visual = VisualElement(
                        element_type=self._map_element_type(element.type),
                        source_format=DocumentType.PDF,
                        content_hash=VisualElement.create_hash(img_bytes),
                        page_or_slide=page_num,
                        bounding_box=bbox,  # Original 0-500 scale
                        raw_data=img_bytes,  # Image bytes included!
                        analysis_metadata={
                            "caption": getattr(element, 'caption', ''),
                            "extracted_by": "docling_improved",
                            "element_type": element.type
                        }
                    )
                    visual_elements.append(visual)
                    
                    logger.debug(
                        f"Extracted {element.type} on page {page_num}: "
                        f"bbox={bbox}, size={len(img_bytes)} bytes"
                    )
        
        return visual_elements
    
    def _get_element_bbox(self, element: Any) -> Optional[List[int]]:
        """Extract bbox from Docling element"""
        # Try different attributes
        if hasattr(element, 'bbox'):
            bbox = element.bbox
            if hasattr(bbox, 'x0'):
                return [int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)]
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return [int(x) for x in bbox[:4]]
        
        # Try parsing from string representation
        element_str = str(element)
        loc_match = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', element_str)
        if loc_match:
            return [int(loc_match.group(i)) for i in range(1, 5)]
        
        return None
    
    def _map_element_type(self, docling_type: str) -> VisualElementType:
        """Map Docling element type to our VisualElementType"""
        mapping = {
            'picture': VisualElementType.IMAGE,
            'image': VisualElementType.IMAGE,
            'figure': VisualElementType.FIGURE,
            'chart': VisualElementType.CHART,
            'diagram': VisualElementType.DIAGRAM,
            'graph': VisualElementType.GRAPH,
            'table': VisualElementType.TABLE,
            'formula': VisualElementType.FORMULA
        }
        return mapping.get(docling_type, VisualElementType.UNKNOWN_VISUAL)
    
    def _create_page_from_docling(
        self, 
        doc: Any,  # DoclingDocument
        page_num: int,
        visual_elements: List[VisualElement]
    ) -> SmolDoclingPage:
        """Create SmolDoclingPage from DoclingDocument"""
        # Extract text content
        text_parts = []
        tables = []
        formulas = []
        
        for element in doc.elements:
            if element.type == 'text':
                text_parts.append(self._get_element_text(element))
            elif element.type == 'table':
                tables.append({
                    "content": self._convert_table_to_text(element),
                    "format": "text"
                })
            elif element.type == 'formula':
                formulas.append({
                    "latex": self._get_element_text(element),
                    "type": "formula"
                })
        
        # Create images list for backward compatibility
        images = []
        for ve in visual_elements:
            if ve.element_type in [VisualElementType.IMAGE, VisualElementType.FIGURE]:
                images.append({
                    "bbox": ve.bounding_box,
                    "caption": ve.analysis_metadata.get("caption", ""),
                    "has_image_data": ve.raw_data is not None
                })
        
        return SmolDoclingPage(
            page_number=page_num,
            text="\n\n".join(text_parts),
            tables=tables,
            images=images,
            formulas=formulas,
            layout_info={
                "source": "docling",
                "visual_elements_extracted": len(visual_elements)
            },
            confidence_score=1.0
        )
    
    def _get_element_text(self, element: Any) -> str:
        """Extract text from element"""
        if hasattr(element, 'text'):
            return element.text
        elif hasattr(element, 'content'):
            return element.content
        return str(element)
    
    def _convert_table_to_text(self, element: Any) -> str:
        """Convert table to text"""
        if hasattr(element, 'to_markdown'):
            return element.to_markdown()
        elif hasattr(element, 'to_text'):
            return element.to_text()
        return str(element)


# Comparison with current approach:
"""
CURRENT APPROACH:
1. SmolDocling generates DocTags with bbox
2. parse_model_output() extracts bbox as metadata
3. HybridPDFParser later calls _extract_image_bytes() using bbox

IMPROVED APPROACH:
1. SmolDocling generates DocTags with bbox
2. Docling parses DocTags into structured elements
3. We extract images IMMEDIATELY while we have both:
   - The DoclingDocument with bbox info
   - The PDF document for image extraction
4. VisualElements are created with raw_data already populated

ADVANTAGES:
- Single pass through the document
- No need to store bbox and extract later
- Docling can help identify image regions better
- More efficient memory usage
- Cleaner architecture
"""