"""Enhanced context mapping for visual elements"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from plugins.parsers.base_parser import Segment, VisualElement

logger = logging.getLogger(__name__)


@dataclass
class ContextMapping:
    """Mapping between visual elements and their textual context"""
    visual_element_hash: str
    related_segments: List[str]  # Segment indices or identifiers
    context_type: str  # "embedded", "adjacent", "page", "section"
    confidence: float  # How confident we are about this mapping
    proximity_score: float  # How close the visual is to the text
    contextual_hints: List[str]  # Text hints that support this mapping


class ContextMapper:
    """
    Enhanced context mapping for visual elements
    
    Provides better association between visual elements and their
    surrounding textual context across different document types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.config.setdefault("proximity_threshold", 0.7)
        self.config.setdefault("context_window_size", 3)  # Number of segments to consider
        self.config.setdefault("keyword_matching", True)
        
    def map_visual_to_context(
        self, 
        visual_elements: List[VisualElement], 
        segments: List[Segment],
        document_type: str
    ) -> List[ContextMapping]:
        """
        Create enhanced mappings between visual elements and their context
        
        Args:
            visual_elements: List of visual elements from document
            segments: List of text segments from document
            document_type: Type of document (PDF, DOCX, etc.)
            
        Returns:
            List of context mappings
        """
        mappings = []
        
        for visual in visual_elements:
            try:
                # Get potential context segments
                candidate_segments = self._get_candidate_segments(visual, segments, document_type)
                
                # Score each candidate segment
                scored_segments = []
                for segment in candidate_segments:
                    score = self._calculate_context_score(visual, segment, document_type)
                    if score > self.config["proximity_threshold"]:
                        scored_segments.append((segment, score))
                
                # Sort by score and create mapping
                scored_segments.sort(key=lambda x: x[1], reverse=True)
                
                if scored_segments:
                    mapping = self._create_context_mapping(visual, scored_segments, document_type)
                    mappings.append(mapping)
                    
            except Exception as e:
                logger.warning(f"Failed to map context for visual {visual.content_hash}: {e}")
                continue
        
        return mappings
    
    def _get_candidate_segments(
        self, 
        visual: VisualElement, 
        segments: List[Segment], 
        document_type: str
    ) -> List[Segment]:
        """Get potential context segments for a visual element"""
        candidates = []
        
        if document_type == "PDF":
            # For PDF: segments on same page + adjacent pages
            target_page = visual.page_or_slide
            if target_page:
                for segment in segments:
                    if segment.page_number:
                        page_diff = abs(segment.page_number - target_page)
                        if page_diff <= 1:  # Same page or adjacent
                            candidates.append(segment)
        
        elif document_type == "DOCX":
            # For DOCX: use existing segment reference + surrounding paragraphs
            if visual.segment_reference:
                # Extract paragraph index from reference
                try:
                    para_idx = int(visual.segment_reference.split("_")[1])
                    window_size = self.config["context_window_size"]
                    
                    for segment in segments:
                        seg_idx = segment.metadata.get("paragraph_index", -1)
                        if seg_idx >= 0 and abs(seg_idx - para_idx) <= window_size:
                            candidates.append(segment)
                except (ValueError, IndexError):
                    pass
        
        elif document_type == "XLSX":
            # For XLSX: segments from same sheet
            if visual.segment_reference:
                sheet_name = visual.segment_reference.replace("sheet_", "")
                for segment in segments:
                    if segment.metadata.get("sheet_name") == sheet_name:
                        candidates.append(segment)
        
        elif document_type == "PPTX":
            # For PPTX: segments from same slide
            target_slide = visual.page_or_slide
            if target_slide:
                for segment in segments:
                    if segment.page_number == target_slide:
                        candidates.append(segment)
        
        return candidates
    
    def _calculate_context_score(
        self, 
        visual: VisualElement, 
        segment: Segment, 
        document_type: str
    ) -> float:
        """Calculate how likely a segment is to be the context for a visual element"""
        score = 0.0
        
        # Base score from proximity
        proximity_score = self._calculate_proximity_score(visual, segment, document_type)
        score += proximity_score * 0.4
        
        # Keyword matching score
        if self.config["keyword_matching"]:
            keyword_score = self._calculate_keyword_score(visual, segment)
            score += keyword_score * 0.3
        
        # Segment type relevance
        type_score = self._calculate_type_relevance(visual, segment)
        score += type_score * 0.2
        
        # Visual reference bonus
        if visual.content_hash in segment.visual_references:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_proximity_score(
        self, 
        visual: VisualElement, 
        segment: Segment, 
        document_type: str
    ) -> float:
        """Calculate proximity score based on position"""
        if document_type == "PDF":
            # Same page = high score, adjacent page = medium score
            if visual.page_or_slide == segment.page_number:
                return 1.0
            elif abs(visual.page_or_slide - segment.page_number) == 1:
                return 0.6
            else:
                return 0.0
        
        elif document_type == "DOCX":
            # Use paragraph indices for proximity
            if visual.segment_reference:
                try:
                    visual_para = int(visual.segment_reference.split("_")[1])
                    segment_para = segment.metadata.get("paragraph_index", -1)
                    
                    if segment_para >= 0:
                        distance = abs(visual_para - segment_para)
                        if distance == 0:
                            return 1.0
                        elif distance <= 2:
                            return 0.8
                        elif distance <= 5:
                            return 0.4
                        else:
                            return 0.1
                except (ValueError, IndexError):
                    pass
        
        elif document_type == "PPTX":
            # Same slide = high score
            if visual.page_or_slide == segment.page_number:
                return 1.0
            else:
                return 0.0
        
        return 0.5  # Default moderate score
    
    def _calculate_keyword_score(self, visual: VisualElement, segment: Segment) -> float:
        """Calculate score based on keyword matching"""
        score = 0.0
        
        # Keywords that suggest visual references
        visual_keywords = [
            "figure", "chart", "graph", "diagram", "image", "picture",
            "table", "screenshot", "illustration", "plot", "map",
            "siehe", "abbildung", "tabelle", "grafik", "diagramm",
            "darstellung", "schaubild", "übersicht"
        ]
        
        segment_text = segment.content.lower()
        
        # Check for visual keywords
        for keyword in visual_keywords:
            if keyword in segment_text:
                score += 0.2
        
        # Check for number references (Figure 1, Table 2, etc.)
        import re
        number_patterns = [
            r'\b(figure|fig|table|chart|graph)\s*\d+',
            r'\b(abbildung|abb|tabelle|graf)\s*\d+',
            r'\b(siehe|vgl)\s+(abbildung|tabelle|grafik)'
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, segment_text):
                score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_type_relevance(self, visual: VisualElement, segment: Segment) -> float:
        """Calculate relevance based on segment and visual types"""
        # Caption segments are highly relevant
        if segment.segment_type in ["caption", "image_caption"]:
            return 1.0
        
        # Heading segments are moderately relevant
        if segment.segment_type in ["heading", "title"]:
            return 0.7
        
        # Table segments are relevant for table images
        if (segment.segment_type == "table" and 
            visual.element_type.value in ["table_image", "chart"]):
            return 0.8
        
        # Regular text segments
        if segment.segment_type in ["text", "paragraph"]:
            return 0.5
        
        return 0.3
    
    def _create_context_mapping(
        self, 
        visual: VisualElement, 
        scored_segments: List[Tuple[Segment, float]], 
        document_type: str
    ) -> ContextMapping:
        """Create a context mapping from scored segments"""
        
        # Get the best segments (top 3 or those above threshold)
        best_segments = scored_segments[:3]
        high_score_segments = [s for s, score in best_segments if score > 0.8]
        
        # Determine context type
        context_type = "adjacent"
        if high_score_segments:
            # Check if any segment has direct reference
            for segment, score in best_segments:
                if visual.content_hash in segment.visual_references:
                    context_type = "embedded"
                    break
                elif score > 0.9:
                    context_type = "direct"
                    break
        
        # Extract contextual hints
        contextual_hints = []
        for segment, score in best_segments[:2]:  # Top 2 segments
            if score > 0.7:
                # Extract key phrases that might reference the visual
                words = segment.content.split()[:50]  # First 50 words
                contextual_hints.extend([
                    phrase for phrase in words 
                    if any(keyword in phrase.lower() for keyword in 
                           ["figure", "chart", "table", "image", "abbildung", "tabelle"])
                ])
        
        # Calculate overall confidence
        avg_score = sum(score for _, score in best_segments) / len(best_segments)
        confidence = min(avg_score + (0.1 if context_type == "embedded" else 0.0), 1.0)
        
        return ContextMapping(
            visual_element_hash=visual.content_hash,
            related_segments=[str(segment.segment_index) for segment, _ in best_segments],
            context_type=context_type,
            confidence=confidence,
            proximity_score=best_segments[0][1] if best_segments else 0.0,
            contextual_hints=contextual_hints[:5]  # Top 5 hints
        )
    
    def apply_context_mappings(
        self, 
        segments: List[Segment], 
        mappings: List[ContextMapping]
    ) -> List[Segment]:
        """Apply context mappings to update segment visual references"""
        
        # Create mapping lookup
        mapping_dict = {}
        for mapping in mappings:
            for segment_idx in mapping.related_segments:
                if segment_idx not in mapping_dict:
                    mapping_dict[segment_idx] = []
                mapping_dict[segment_idx].append(mapping)
        
        # Update segments
        updated_segments = []
        for segment in segments:
            segment_key = str(segment.segment_index)
            
            if segment_key in mapping_dict:
                # Update visual references based on context mappings
                enhanced_refs = set(segment.visual_references)
                
                for mapping in mapping_dict[segment_key]:
                    if mapping.confidence > 0.7:  # Only high-confidence mappings
                        enhanced_refs.add(mapping.visual_element_hash)
                
                # Update segment
                segment.visual_references = list(enhanced_refs)
                
                # Add context mapping info to metadata
                segment.metadata["context_mappings"] = [
                    {
                        "visual_hash": m.visual_element_hash,
                        "context_type": m.context_type,
                        "confidence": m.confidence,
                        "hints": m.contextual_hints
                    }
                    for m in mapping_dict[segment_key]
                ]
            
            updated_segments.append(segment)
        
        return updated_segments