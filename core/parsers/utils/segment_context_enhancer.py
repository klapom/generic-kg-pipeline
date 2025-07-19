"""
Segment Context Enhancer - Adds contextual information to important segments
Focuses on tables, lists, and visual elements for better VLM analysis
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import re

from core.parsers.interfaces.data_models import (
    Segment, SegmentType, TextSubtype, DocumentMetadata
)

logger = logging.getLogger(__name__)


class SegmentContextEnhancer:
    """Enhances segments with contextual information for better understanding"""
    
    # Define which segment types benefit from context
    CONTEXT_WORTHY_TYPES = {
        (SegmentType.TABLE, None),                    # All tables need context
        (SegmentType.TEXT, TextSubtype.LIST.value),  # Lists need context
        (SegmentType.VISUAL, None),                  # Visual elements benefit from context
    }
    
    # Heading subtypes to consider as context providers
    HEADING_SUBTYPES = {
        TextSubtype.HEADING_1.value,
        TextSubtype.HEADING_2.value,
        TextSubtype.HEADING_3.value,
        TextSubtype.TITLE.value,
        TextSubtype.SUBTITLE.value
    }
    
    @staticmethod
    def needs_context(segment: Segment) -> bool:
        """Determines if a segment would benefit from context enhancement"""
        # Check if segment type/subtype combination needs context
        for seg_type, seg_subtype in SegmentContextEnhancer.CONTEXT_WORTHY_TYPES:
            if segment.segment_type == seg_type:
                if seg_subtype is None or segment.segment_subtype == seg_subtype:
                    logger.debug(f"Segment {segment.segment_index} ({segment.segment_type}/{segment.segment_subtype}) needs context")
                    return True
        
        # Additional check for lists that might not be properly tagged
        if segment.segment_type == SegmentType.TEXT and segment.content:
            # Check for list patterns
            list_patterns = ["•", "●", "- ", "* ", "1.", "2.", "a)", "b)"]
            for pattern in list_patterns:
                if pattern in segment.content:
                    logger.info(f"Segment {segment.segment_index} has list pattern '{pattern}', marking as needs context")
                    return True
        
        logger.debug(f"Segment {segment.segment_index} ({segment.segment_type}/{segment.segment_subtype}) does not need context")
        return False
    
    @staticmethod
    def enhance_segments(segments: List[Segment], document_metadata: DocumentMetadata) -> None:
        """
        Enhances segments with contextual information
        
        Args:
            segments: List of segments to enhance
            document_metadata: Document metadata for additional context
        """
        logger.info(f"🔍 Starting context enhancement for {len(segments)} segments")
        
        # Build document structure for efficient lookup
        headings_by_page = SegmentContextEnhancer._build_heading_index(segments)
        
        # Count enhanced segments
        enhanced_count = 0
        
        # Process each segment
        for i, segment in enumerate(segments):
            if not SegmentContextEnhancer.needs_context(segment):
                continue
            
            # Initialize context if not exists
            if "context" not in segment.metadata:
                segment.metadata["context"] = {}
            
            # Find nearest heading
            nearest_heading = SegmentContextEnhancer._find_nearest_heading(i, segments)
            if nearest_heading:
                segment.metadata["context"]["nearest_heading"] = nearest_heading
                logger.debug(f"  Found heading for segment {i}: '{nearest_heading[:50]}...'")
            
            # Add document-level context
            if document_metadata.title:
                segment.metadata["context"]["document_title"] = document_metadata.title
            
            # Determine document type from metadata or content
            doc_type = SegmentContextEnhancer._classify_document_type(document_metadata, segments)
            if doc_type:
                segment.metadata["context"]["document_type"] = doc_type
            
            # Add position context
            segment.metadata["context"]["position"] = {
                "segment_index": i,
                "total_segments": len(segments),
                "page": segment.page_number,
                "relative_position": SegmentContextEnhancer._get_relative_position(i, segments, segment.page_number)
            }
            
            # Type-specific context
            if segment.segment_type == SegmentType.TABLE:
                table_ref = SegmentContextEnhancer._find_table_reference(i, segments)
                if table_ref:
                    segment.metadata["context"]["table_reference"] = table_ref
                    logger.debug(f"  Found table reference: '{table_ref}'")
                    
            elif segment.segment_subtype == TextSubtype.LIST.value or (
                segment.segment_type == SegmentType.TEXT and 
                any(pattern in segment.content for pattern in ["•", "●", "- ", "* "])
            ):
                list_intro = SegmentContextEnhancer._find_list_introduction(i, segments)
                if list_intro:
                    segment.metadata["context"]["list_introduction"] = list_intro
                    logger.debug(f"  Found list introduction: '{list_intro[:50]}...'")
            
            enhanced_count += 1
        
        logger.info(f"✅ Enhanced {enhanced_count} segments with context")
    
    @staticmethod
    def _build_heading_index(segments: List[Segment]) -> Dict[int, List[Tuple[int, str]]]:
        """Build index of headings by page for quick lookup"""
        headings_by_page = defaultdict(list)
        
        for i, segment in enumerate(segments):
            if segment.segment_subtype in SegmentContextEnhancer.HEADING_SUBTYPES:
                page = segment.page_number or 0
                headings_by_page[page].append((i, segment.content))
                
        return dict(headings_by_page)
    
    @staticmethod
    def _find_nearest_heading(target_index: int, segments: List[Segment]) -> Optional[str]:
        """Find the nearest heading before the target segment"""
        # Search backwards from target
        for i in range(target_index - 1, -1, -1):
            if segments[i].segment_subtype in SegmentContextEnhancer.HEADING_SUBTYPES:
                return segments[i].content
        
        return None
    
    @staticmethod
    def _classify_document_type(metadata: DocumentMetadata, segments: List[Segment]) -> Optional[str]:
        """Classify document type based on metadata and content"""
        # Check metadata first
        if metadata.custom_metadata.get("document_type"):
            return metadata.custom_metadata["document_type"]
        
        # Analyze title and early content
        title_lower = (metadata.title or "").lower()
        
        # Common document type patterns
        if any(term in title_lower for term in ["specification", "technical", "datasheet"]):
            return "Technical Specification"
        elif any(term in title_lower for term in ["manual", "guide", "instruction"]):
            return "Manual"
        elif any(term in title_lower for term in ["report", "analysis", "study"]):
            return "Report"
        elif any(term in title_lower for term in ["presentation", "slides"]):
            return "Presentation"
        
        # Check first few headings
        heading_texts = []
        for segment in segments[:20]:  # Check first 20 segments
            if segment.segment_subtype in SegmentContextEnhancer.HEADING_SUBTYPES:
                heading_texts.append(segment.content.lower())
        
        combined_headings = " ".join(heading_texts)
        
        if any(term in combined_headings for term in ["motor", "engine", "performance", "specification"]):
            return "Technical Specification"
        elif any(term in combined_headings for term in ["price", "cost", "model", "variant"]):
            return "Product Catalog"
        
        return "Document"  # Generic fallback
    
    @staticmethod
    def _get_relative_position(index: int, segments: List[Segment], page_num: Optional[int]) -> str:
        """Determine relative position on page"""
        if page_num is None:
            return "unknown"
        
        # Find all segments on same page
        page_segments = [(i, s) for i, s in enumerate(segments) if s.page_number == page_num]
        
        if not page_segments:
            return "unknown"
        
        # Find position of current segment in page
        position_in_page = next((i for i, (idx, _) in enumerate(page_segments) if idx == index), 0)
        
        relative_pos = position_in_page / len(page_segments)
        
        if relative_pos < 0.33:
            return "top"
        elif relative_pos < 0.67:
            return "middle"
        else:
            return "bottom"
    
    @staticmethod
    def _find_table_reference(table_index: int, segments: List[Segment]) -> Optional[str]:
        """Find reference to table in preceding text"""
        # Common patterns for table references
        table_patterns = [
            r'[Tt]abelle?\s*\d+',  # German: Tabelle 1
            r'[Tt]able\s*\d+',     # English: Table 1
            r'[Ff]olgende\s+[Tt]abelle',  # "folgende Tabelle"
            r'[Ff]ollowing\s+[Tt]able',   # "following table"
            r'[Nn]achstehende\s+[Tt]abelle',  # "nachstehende Tabelle"
        ]
        
        # Search in previous 3 segments
        for i in range(max(0, table_index - 3), table_index):
            segment_text = segments[i].content
            
            for pattern in table_patterns:
                match = re.search(pattern, segment_text)
                if match:
                    # Extract surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(segment_text), match.end() + 50)
                    return segment_text[start:end].strip()
        
        return None
    
    @staticmethod
    def _find_list_introduction(list_index: int, segments: List[Segment]) -> Optional[str]:
        """Find introductory text for a list"""
        # Patterns that typically introduce lists
        intro_patterns = [
            r'[Ff]olgende\s+\w+:',  # "Folgende Punkte:"
            r'[Ff]ollowing\s+\w+:',  # "Following items:"
            r'[Dd]ie\s+\w+\s+umfassen:',  # "Die Features umfassen:"
            r'[Ii]ncluding:',
            r'[Ww]ie\s+folgt:',  # "wie folgt:"
            r':$',  # Ends with colon
        ]
        
        # Check previous segment
        if list_index > 0:
            prev_segment = segments[list_index - 1]
            prev_text = prev_segment.content.strip()
            
            for pattern in intro_patterns:
                if re.search(pattern, prev_text):
                    # Return last sentence/line
                    sentences = prev_text.split('.')
                    return sentences[-1].strip() if sentences else prev_text
        
        return None
    
    @staticmethod
    def get_context_for_prompt(segment: Segment) -> str:
        """
        Generate a context string suitable for VLM prompts
        
        Args:
            segment: Segment with context metadata
            
        Returns:
            Context string for prompt enhancement
        """
        context_parts = []
        
        context = segment.metadata.get("context", {})
        
        # Add document type if available
        if context.get("document_type"):
            context_parts.append(f"Document type: {context['document_type']}")
        
        # Add nearest heading
        if context.get("nearest_heading"):
            context_parts.append(f"Under section: {context['nearest_heading']}")
        
        # Add specific references
        if context.get("table_reference"):
            context_parts.append(f"Referenced as: {context['table_reference']}")
        elif context.get("list_introduction"):
            context_parts.append(f"Introduced by: {context['list_introduction']}")
        
        # Combine into single context string
        if context_parts:
            return " | ".join(context_parts)
        
        return ""