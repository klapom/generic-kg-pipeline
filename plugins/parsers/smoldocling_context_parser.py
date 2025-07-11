"""Enhanced PDF parsing with SmolDocling structured context mapping"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from plugins.parsers.base_parser import (
    Segment, 
    VisualElement, 
    VisualElementType, 
    DocumentType
)

logger = logging.getLogger(__name__)


@dataclass
class SmolDoclingElement:
    """Represents a structured element from SmolDocling"""
    element_type: str  # "text", "table", "image", "formula"
    content: str
    page_number: int
    position_in_page: int  # Order within the page
    caption: Optional[str] = None
    description: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class SmolDoclingContextParser:
    """
    Parser for SmolDocling structured output with precise context mapping
    
    SmolDocling returns structured data with:
    - Text blocks with position information
    - Images with captions and descriptions
    - Tables with captions
    - Formulas with descriptions
    - All elements have page numbers and positioning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.config.setdefault("merge_adjacent_text", True)
        self.config.setdefault("create_caption_segments", True)
        self.config.setdefault("link_references", True)
        
    def parse_smoldocling_result(self, smoldocling_data: Dict[str, Any]) -> Tuple[List[Segment], List[VisualElement]]:
        """
        Parse SmolDocling structured output into segments and visual elements
        
        Args:
            smoldocling_data: Structured data from SmolDocling
            
        Returns:
            Tuple of (segments, visual_elements) with precise context mapping
        """
        try:
            # Extract structured elements from SmolDocling data
            structured_elements = self._extract_structured_elements(smoldocling_data)
            
            # Create segments from structured elements
            segments = self._create_segments_from_elements(structured_elements)
            
            # Create visual elements from structured elements
            visual_elements = self._create_visual_elements_from_elements(structured_elements)
            
            # Apply context linking between segments and visual elements
            segments, visual_elements = self._apply_context_linking(segments, visual_elements, structured_elements)
            
            logger.info(f"SmolDocling parsing: {len(segments)} segments, {len(visual_elements)} visual elements")
            
            return segments, visual_elements
            
        except Exception as e:
            logger.error(f"SmolDocling context parsing failed: {e}")
            raise
    
    def _extract_structured_elements(self, smoldocling_data: Dict[str, Any]) -> List[SmolDoclingElement]:
        """Extract structured elements from SmolDocling output"""
        elements = []
        
        try:
            pages = smoldocling_data.get("pages", [])
            
            for page_data in pages:
                page_number = page_data.get("page_number", 1)
                position_counter = 0
                
                # Extract text content
                text_content = page_data.get("text", "")
                if text_content:
                    # Split text into logical blocks (paragraphs)
                    text_blocks = self._split_text_into_blocks(text_content)
                    
                    for block in text_blocks:
                        if block.strip():
                            elements.append(SmolDoclingElement(
                                element_type="text",
                                content=block.strip(),
                                page_number=page_number,
                                position_in_page=position_counter,
                                metadata={"source": "text_extraction"}
                            ))
                            position_counter += 1
                
                # Extract tables
                tables = page_data.get("tables", [])
                for table_data in tables:
                    table_content = self._format_table_content(table_data)
                    elements.append(SmolDoclingElement(
                        element_type="table",
                        content=table_content,
                        page_number=page_number,
                        position_in_page=position_counter,
                        caption=table_data.get("caption"),
                        metadata={
                            "source": "table_extraction",
                            "headers": table_data.get("headers", []),
                            "row_count": len(table_data.get("rows", []))
                        }
                    ))
                    position_counter += 1
                
                # Extract images
                images = page_data.get("images", [])
                for image_data in images:
                    elements.append(SmolDoclingElement(
                        element_type="image",
                        content=image_data.get("description", ""),
                        page_number=page_number,
                        position_in_page=position_counter,
                        caption=image_data.get("caption"),
                        description=image_data.get("description"),
                        metadata={
                            "source": "image_extraction",
                            "image_type": image_data.get("image_type", "figure")
                        }
                    ))
                    position_counter += 1
                
                # Extract formulas
                formulas = page_data.get("formulas", [])
                for formula_data in formulas:
                    elements.append(SmolDoclingElement(
                        element_type="formula",
                        content=formula_data.get("latex", ""),
                        page_number=page_number,
                        position_in_page=position_counter,
                        description=formula_data.get("description"),
                        metadata={
                            "source": "formula_extraction",
                            "latex": formula_data.get("latex")
                        }
                    ))
                    position_counter += 1
            
            # Sort elements by page and position
            elements.sort(key=lambda x: (x.page_number, x.position_in_page))
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to extract structured elements: {e}")
            return []
    
    def _split_text_into_blocks(self, text: str) -> List[str]:
        """Split text into logical blocks (paragraphs)"""
        # Split by double newlines for paragraphs
        blocks = text.split('\n\n')
        
        # Further split very long blocks
        final_blocks = []
        for block in blocks:
            if len(block) > 1000:  # Very long block
                # Split by single newlines but keep related sentences together
                sentences = re.split(r'(?<=[.!?])\s+', block)
                current_block = ""
                
                for sentence in sentences:
                    if len(current_block + sentence) > 500:
                        if current_block:
                            final_blocks.append(current_block.strip())
                            current_block = sentence
                        else:
                            final_blocks.append(sentence.strip())
                    else:
                        current_block += " " + sentence if current_block else sentence
                
                if current_block:
                    final_blocks.append(current_block.strip())
            else:
                final_blocks.append(block)
        
        return [block for block in final_blocks if block.strip()]
    
    def _format_table_content(self, table_data: Dict[str, Any]) -> str:
        """Format table data as text"""
        lines = []
        
        # Add caption if available
        if table_data.get("caption"):
            lines.append(f"Table: {table_data['caption']}")
            lines.append("")
        
        # Add headers
        headers = table_data.get("headers", [])
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * len(" | ".join(headers)))
        
        # Add rows
        rows = table_data.get("rows", [])
        for row in rows:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        
        return "\n".join(lines)
    
    def _create_segments_from_elements(self, elements: List[SmolDoclingElement]) -> List[Segment]:
        """Create segments from structured elements"""
        segments = []
        segment_index = 0
        
        for element in elements:
            # Create main content segment
            if element.element_type == "text":
                segment_type = self._determine_text_segment_type(element.content)
            else:
                segment_type = element.element_type
            
            segment = Segment(
                content=element.content,
                page_number=element.page_number,
                segment_index=segment_index,
                segment_type=segment_type,
                metadata={
                    "smoldocling_element_type": element.element_type,
                    "position_in_page": element.position_in_page,
                    "element_id": f"{element.element_type}_{element.page_number}_{element.position_in_page}"
                }
            )
            
            # Add element-specific metadata
            if element.metadata:
                segment.metadata.update(element.metadata)
            
            segments.append(segment)
            segment_index += 1
            
            # Create separate caption segment if available
            if (element.caption and 
                self.config.get("create_caption_segments", True) and
                element.element_type in ["table", "image"]):
                
                caption_segment = Segment(
                    content=element.caption,
                    page_number=element.page_number,
                    segment_index=segment_index,
                    segment_type="caption",
                    metadata={
                        "smoldocling_element_type": "caption",
                        "parent_element_type": element.element_type,
                        "parent_element_id": f"{element.element_type}_{element.page_number}_{element.position_in_page}",
                        "caption_for": element.element_type
                    }
                )
                
                segments.append(caption_segment)
                segment_index += 1
        
        return segments
    
    def _create_visual_elements_from_elements(self, elements: List[SmolDoclingElement]) -> List[VisualElement]:
        """Create visual elements from structured elements"""
        visual_elements = []
        
        for element in elements:
            if element.element_type in ["image", "table", "formula"]:
                # Determine visual element type
                if element.element_type == "image":
                    img_type = self._determine_image_type(element.metadata.get("image_type", "figure"))
                elif element.element_type == "table":
                    img_type = VisualElementType.TABLE_IMAGE
                elif element.element_type == "formula":
                    img_type = VisualElementType.DIAGRAM
                else:
                    img_type = VisualElementType.UNKNOWN_VISUAL
                
                # Create content for hashing
                content_for_hash = f"{element.element_type}_{element.page_number}_{element.position_in_page}_{element.content}"
                
                visual_element = VisualElement(
                    element_type=img_type,
                    source_format=DocumentType.PDF,
                    content_hash=VisualElement.create_hash(content_for_hash.encode('utf-8')),
                    vlm_description=element.description,  # SmolDocling already provides description
                    page_or_slide=element.page_number,
                    segment_reference=f"{element.element_type}_{element.page_number}_{element.position_in_page}",
                    file_extension="txt" if element.element_type == "formula" else "png",
                    confidence=0.9,  # High confidence since it's from SmolDocling
                    analysis_metadata={
                        "smoldocling_element_type": element.element_type,
                        "position_in_page": element.position_in_page,
                        "caption": element.caption,
                        "smoldocling_description": element.description,
                        "extraction_method": "smoldocling_structured"
                    }
                )
                
                # Add element-specific data
                if element.metadata:
                    visual_element.analysis_metadata.update(element.metadata)
                
                visual_elements.append(visual_element)
        
        return visual_elements
    
    def _determine_text_segment_type(self, content: str) -> str:
        """Determine the type of text segment"""
        content_lower = content.lower().strip()
        
        # Heading patterns
        if (len(content) < 100 and 
            (content.isupper() or 
             re.match(r'^\d+\.?\s+[A-Z]', content) or
             re.match(r'^[A-Z][^.]*$', content))):
            return "heading"
        
        # Caption patterns
        if (content_lower.startswith(('figure', 'table', 'abbildung', 'tabelle')) or
            re.match(r'^(fig|tab|abb)\.\s*\d+', content_lower)):
            return "caption"
        
        # List patterns
        if re.match(r'^\s*[-•]\s+', content) or re.match(r'^\s*\d+\.\s+', content):
            return "list"
        
        return "text"
    
    def _determine_image_type(self, image_type_hint: str) -> VisualElementType:
        """Determine VisualElementType from SmolDocling image type hint"""
        type_map = {
            "figure": VisualElementType.IMAGE,
            "chart": VisualElementType.CHART,
            "graph": VisualElementType.GRAPH,
            "diagram": VisualElementType.DIAGRAM,
            "table": VisualElementType.TABLE_IMAGE,
            "screenshot": VisualElementType.SCREENSHOT,
            "map": VisualElementType.MAP
        }
        
        return type_map.get(image_type_hint.lower(), VisualElementType.IMAGE)
    
    def _apply_context_linking(
        self, 
        segments: List[Segment], 
        visual_elements: List[VisualElement],
        structured_elements: List[SmolDoclingElement]
    ) -> Tuple[List[Segment], List[VisualElement]]:
        """Apply precise context linking between segments and visual elements"""
        
        if not self.config.get("link_references", True):
            return segments, visual_elements
        
        try:
            # Create lookup dictionaries
            element_lookup = {
                f"{elem.element_type}_{elem.page_number}_{elem.position_in_page}": elem
                for elem in structured_elements
            }
            
            visual_lookup = {ve.segment_reference: ve for ve in visual_elements}
            
            # Process each segment
            for segment in segments:
                element_id = segment.metadata.get("element_id")
                
                if element_id and element_id in element_lookup:
                    element = element_lookup[element_id]
                    
                    # Link to visual elements
                    visual_refs = []
                    
                    # 1. Direct reference (same element)
                    if element_id in visual_lookup:
                        visual_refs.append(visual_lookup[element_id].content_hash)
                    
                    # 2. Find references in nearby elements
                    nearby_visual_refs = self._find_nearby_visual_references(
                        element, structured_elements, visual_lookup
                    )
                    visual_refs.extend(nearby_visual_refs)
                    
                    # 3. Find textual references (e.g., "see Figure 1")
                    textual_refs = self._find_textual_references(
                        segment.content, visual_elements, element.page_number
                    )
                    visual_refs.extend(textual_refs)
                    
                    # Update segment with visual references
                    segment.visual_references = list(set(visual_refs))
                    
                    # Add context information to metadata
                    segment.metadata["context_linking"] = {
                        "direct_references": 1 if element_id in visual_lookup else 0,
                        "nearby_references": len(nearby_visual_refs),
                        "textual_references": len(textual_refs),
                        "total_visual_links": len(segment.visual_references)
                    }
            
            return segments, visual_elements
            
        except Exception as e:
            logger.error(f"Context linking failed: {e}")
            return segments, visual_elements
    
    def _find_nearby_visual_references(
        self, 
        element: SmolDoclingElement, 
        all_elements: List[SmolDoclingElement],
        visual_lookup: Dict[str, VisualElement]
    ) -> List[str]:
        """Find visual elements near the current element"""
        visual_refs = []
        
        # Look for visual elements on the same page within a small position range
        for other_element in all_elements:
            if (other_element.element_type in ["image", "table", "formula"] and
                other_element.page_number == element.page_number and
                abs(other_element.position_in_page - element.position_in_page) <= 2):
                
                other_id = f"{other_element.element_type}_{other_element.page_number}_{other_element.position_in_page}"
                if other_id in visual_lookup:
                    visual_refs.append(visual_lookup[other_id].content_hash)
        
        return visual_refs
    
    def _find_textual_references(
        self, 
        text: str, 
        visual_elements: List[VisualElement], 
        current_page: int
    ) -> List[str]:
        """Find visual elements referenced in text"""
        visual_refs = []
        text_lower = text.lower()
        
        # Common reference patterns
        reference_patterns = [
            r'\b(figure|fig|abbildung|abb)\s*(\d+)',
            r'\b(table|tab|tabelle)\s*(\d+)',
            r'\b(chart|graph|grafik)\s*(\d+)',
            r'\b(diagram|diagramm)\s*(\d+)',
            r'\b(see|siehe)\s+(figure|table|abbildung|tabelle)\s*(\d+)',
        ]
        
        for pattern in reference_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Try to find corresponding visual element
                for ve in visual_elements:
                    if (ve.page_or_slide == current_page and
                        ve.analysis_metadata.get("smoldocling_element_type") in ["image", "table"]):
                        
                        # Check if caption contains the reference number
                        caption = ve.analysis_metadata.get("caption", "")
                        if caption and any(str(num) in caption for num in match if num.isdigit()):
                            visual_refs.append(ve.content_hash)
        
        return visual_refs