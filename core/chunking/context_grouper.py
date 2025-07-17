"""Context grouper for forming logical chunk groups"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.parsers import Document, DocumentType, Segment
from .chunk_models import (
    ContextualChunk,
    ContextGroup,
    ContextGroupType,
    StructureInfo
)

logger = logging.getLogger(__name__)


class ContextGrouper:
    """
    Forms logical context groups from document segments
    
    Groups chunks based on document structure to enable context inheritance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize context grouper with configuration"""
        self.config = config
        self.group_formation_config = config.get("group_formation", {})
        
        logger.info("Initialized ContextGrouper")
    
    def group_chunks(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """
        Group chunks into logical context units
        
        Args:
            chunks: List of contextual chunks
            document: Source document
            
        Returns:
            List of context groups
        """
        if not chunks:
            return []
        
        try:
            # Determine grouping strategy based on document type
            document_type = document.metadata.document_type
            
            if document_type == DocumentType.PDF:
                groups = self._group_pdf_by_structure(chunks, document)
            elif document_type == DocumentType.DOCX:
                groups = self._group_docx_by_headings(chunks, document)
            elif document_type == DocumentType.XLSX:
                groups = self._group_xlsx_by_sheets(chunks, document)
            elif document_type == DocumentType.PPTX:
                groups = self._group_pptx_by_topics(chunks, document)
            else:
                # Fallback to simple sequential grouping
                groups = self._group_sequential(chunks, document)
            
            # Apply group size constraints
            groups = self._apply_group_constraints(groups, document_type)
            
            # Update chunk positions within groups
            self._update_chunk_positions(groups)
            
            logger.info(f"Created {len(groups)} context groups for document {document.metadata.title}")
            return groups
            
        except Exception as e:
            logger.error(f"Context grouping failed: {e}")
            # Fallback to single group
            return self._create_fallback_group(chunks, document)
    
    def _group_pdf_by_structure(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Group PDF chunks by structural elements (sections, chapters)"""
        groups = []
        current_group_chunks = []
        current_section = None
        
        pdf_config = self.group_formation_config.get("pdf", {})
        
        for chunk in chunks:
            # Detect section changes
            chunk_section = self._detect_pdf_section(chunk, document)
            
            if chunk_section and chunk_section != current_section:
                # Start new group at section boundary
                if current_group_chunks:
                    group = self._create_context_group(
                        chunks=current_group_chunks,
                        document=document,
                        group_type=ContextGroupType.SECTION,
                        group_metadata={"section": current_section}
                    )
                    groups.append(group)
                
                current_group_chunks = [chunk]
                current_section = chunk_section
            else:
                current_group_chunks.append(chunk)
        
        # Add final group
        if current_group_chunks:
            group = self._create_context_group(
                chunks=current_group_chunks,
                document=document,
                group_type=ContextGroupType.SECTION,
                group_metadata={"section": current_section}
            )
            groups.append(group)
        
        return groups
    
    def _detect_pdf_section(self, chunk: ContextualChunk, document: Document) -> Optional[str]:
        """Detect section markers in PDF chunk"""
        try:
            # Check if chunk contains heading segments
            heading_segments = []
            for seg_ref in chunk.segment_references:
                seg_idx = int(seg_ref.split("_")[1])
                if seg_idx < len(document.segments):
                    segment = document.segments[seg_idx]
                    if segment.segment_type == "heading":
                        heading_segments.append(segment)
            
            if heading_segments:
                return heading_segments[0].content.strip()
            
            # Check for SmolDocling structure markers
            if chunk.processing_metadata.get("smoldocling_element_type") == "heading":
                return chunk.content.split('\n')[0].strip()
            
            # Pattern-based detection
            lines = chunk.content.split('\n')
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                
                # Numbered sections: "1. Introduction", "2.1 Methods"
                if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', line):
                    return line
                
                # All caps headings
                if line.isupper() and len(line) < 100:
                    return line
                
                # Roman numerals: "I. Introduction"
                if re.match(r'^[IVX]+\.\s+[A-Z]', line):
                    return line
            
            return None
            
        except Exception as e:
            logger.debug(f"Section detection failed for chunk {chunk.chunk_id}: {e}")
            return None
    
    def _group_docx_by_headings(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Group DOCX chunks by heading hierarchy"""
        groups = []
        current_group_chunks = []
        current_heading = None
        current_level = 0
        
        for chunk in chunks:
            # Detect heading changes
            heading_info = self._detect_docx_heading(chunk, document)
            
            if heading_info:
                heading_text, heading_level = heading_info
                
                # Start new group if heading level is equal or higher
                if heading_level <= current_level or current_heading is None:
                    if current_group_chunks:
                        group = self._create_context_group(
                            chunks=current_group_chunks,
                            document=document,
                            group_type=ContextGroupType.HEADING_HIERARCHY,
                            group_metadata={
                                "heading": current_heading,
                                "level": current_level
                            }
                        )
                        groups.append(group)
                    
                    current_group_chunks = [chunk]
                    current_heading = heading_text
                    current_level = heading_level
                else:
                    current_group_chunks.append(chunk)
            else:
                current_group_chunks.append(chunk)
        
        # Add final group
        if current_group_chunks:
            group = self._create_context_group(
                chunks=current_group_chunks,
                document=document,
                group_type=ContextGroupType.HEADING_HIERARCHY,
                group_metadata={
                    "heading": current_heading,
                    "level": current_level
                }
            )
            groups.append(group)
        
        return groups
    
    def _detect_docx_heading(self, chunk: ContextualChunk, document: Document) -> Optional[Tuple[str, int]]:
        """Detect heading and level in DOCX chunk"""
        try:
            # Check segment metadata for heading styles
            for seg_ref in chunk.segment_references:
                seg_idx = int(seg_ref.split("_")[1])
                if seg_idx < len(document.segments):
                    segment = document.segments[seg_idx]
                    
                    if segment.segment_type == "heading":
                        # Try to extract heading level from style
                        style = segment.metadata.get("style", "")
                        if "Heading" in style:
                            level_match = re.search(r'Heading (\d+)', style)
                            level = int(level_match.group(1)) if level_match else 1
                            return segment.content.strip(), level
                        else:
                            return segment.content.strip(), 1
            
            return None
            
        except Exception as e:
            logger.debug(f"Heading detection failed for chunk {chunk.chunk_id}: {e}")
            return None
    
    def _group_xlsx_by_sheets(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Group XLSX chunks by sheets"""
        groups = []
        sheet_groups = {}
        
        for chunk in chunks:
            # Detect sheet name
            sheet_name = self._detect_xlsx_sheet(chunk, document)
            
            if sheet_name not in sheet_groups:
                sheet_groups[sheet_name] = []
            
            sheet_groups[sheet_name].append(chunk)
        
        # Create groups from sheet collections
        for sheet_name, sheet_chunks in sheet_groups.items():
            group = self._create_context_group(
                chunks=sheet_chunks,
                document=document,
                group_type=ContextGroupType.SHEET_GROUP,
                group_metadata={"sheet_name": sheet_name}
            )
            groups.append(group)
        
        return groups
    
    def _detect_xlsx_sheet(self, chunk: ContextualChunk, document: Document) -> str:
        """Detect sheet name from XLSX chunk"""
        try:
            # Check segment metadata for sheet name
            for seg_ref in chunk.segment_references:
                seg_idx = int(seg_ref.split("_")[1])
                if seg_idx < len(document.segments):
                    segment = document.segments[seg_idx]
                    sheet_name = segment.metadata.get("sheet_name")
                    if sheet_name:
                        return sheet_name
            
            # Fallback to default sheet name
            return "Sheet1"
            
        except Exception:
            return "Unknown"
    
    def _group_pptx_by_topics(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Group PPTX chunks by topic coherence"""
        groups = []
        current_group_chunks = []
        current_topic = None
        
        for chunk in chunks:
            # Detect topic changes (simplified approach)
            chunk_topic = self._detect_pptx_topic(chunk, document)
            
            if chunk_topic and chunk_topic != current_topic:
                # Start new group
                if current_group_chunks:
                    group = self._create_context_group(
                        chunks=current_group_chunks,
                        document=document,
                        group_type=ContextGroupType.SLIDE_SET,
                        group_metadata={"topic": current_topic}
                    )
                    groups.append(group)
                
                current_group_chunks = [chunk]
                current_topic = chunk_topic
            else:
                current_group_chunks.append(chunk)
        
        # Add final group
        if current_group_chunks:
            group = self._create_context_group(
                chunks=current_group_chunks,
                document=document,
                group_type=ContextGroupType.SLIDE_SET,
                group_metadata={"topic": current_topic}
            )
            groups.append(group)
        
        return groups
    
    def _detect_pptx_topic(self, chunk: ContextualChunk, document: Document) -> Optional[str]:
        """Detect topic from PPTX chunk (simplified)"""
        try:
            # Check if chunk contains slide title
            for seg_ref in chunk.segment_references:
                seg_idx = int(seg_ref.split("_")[1])
                if seg_idx < len(document.segments):
                    segment = document.segments[seg_idx]
                    if segment.segment_type in ["title", "slide"]:
                        return segment.content.strip()
            
            # Extract first line as potential topic
            lines = chunk.content.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('['):
                    return line.strip()
            
            return None
            
        except Exception:
            return None
    
    def _group_sequential(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Fallback sequential grouping"""
        groups = []
        group_size = 6  # Default group size
        
        for i in range(0, len(chunks), group_size):
            group_chunks = chunks[i:i + group_size]
            group = self._create_context_group(
                chunks=group_chunks,
                document=document,
                group_type=ContextGroupType.PAGE_RANGE,
                group_metadata={"sequential_group": i // group_size}
            )
            groups.append(group)
        
        return groups
    
    def _apply_group_constraints(self, groups: List[ContextGroup], document_type: DocumentType) -> List[ContextGroup]:
        """Apply size constraints to groups"""
        doc_config = self.group_formation_config.get(document_type.value, {})
        max_group_size = doc_config.get("max_group_size", 8)
        min_group_size = doc_config.get("min_group_size", 2)
        
        constrained_groups = []
        
        for group in groups:
            if len(group.chunks) > max_group_size:
                # Split large groups
                split_groups = self._split_large_group(group, max_group_size)
                constrained_groups.extend(split_groups)
            elif len(group.chunks) < min_group_size:
                # Merge small groups (simplified approach)
                constrained_groups.append(group)
            else:
                constrained_groups.append(group)
        
        # Merge consecutive small groups
        merged_groups = self._merge_small_groups(constrained_groups, min_group_size)
        
        return merged_groups
    
    def _split_large_group(self, group: ContextGroup, max_size: int) -> List[ContextGroup]:
        """Split a large group into smaller groups"""
        split_groups = []
        chunks = group.chunks
        
        for i in range(0, len(chunks), max_size):
            sub_chunks = chunks[i:i + max_size]
            
            # Create new group with modified ID
            sub_group = ContextGroup(
                group_id=f"{group.group_id}_split_{i // max_size}",
                document_id=group.document_id,
                group_type=group.group_type,
                chunks=sub_chunks,
                structure_info=group.structure_info,
                group_metadata={
                    **group.group_metadata,
                    "split_from": group.group_id,
                    "split_index": i // max_size
                }
            )
            split_groups.append(sub_group)
        
        return split_groups
    
    def _merge_small_groups(self, groups: List[ContextGroup], min_size: int) -> List[ContextGroup]:
        """Merge consecutive small groups"""
        if not groups:
            return groups
        
        merged_groups = []
        i = 0
        
        while i < len(groups):
            current_group = groups[i]
            
            # Check if current group is small and can be merged with next
            if (len(current_group.chunks) < min_size and 
                i + 1 < len(groups) and 
                len(groups[i + 1].chunks) < min_size and
                current_group.group_type == groups[i + 1].group_type):
                
                # Merge with next group
                next_group = groups[i + 1]
                merged_chunks = current_group.chunks + next_group.chunks
                
                merged_group = ContextGroup(
                    group_id=f"{current_group.group_id}_merged_{next_group.group_id}",
                    document_id=current_group.document_id,
                    group_type=current_group.group_type,
                    chunks=merged_chunks,
                    structure_info=current_group.structure_info,
                    group_metadata={
                        **current_group.group_metadata,
                        "merged_from": [current_group.group_id, next_group.group_id]
                    }
                )
                merged_groups.append(merged_group)
                i += 2  # Skip next group as it's been merged
            else:
                merged_groups.append(current_group)
                i += 1
        
        return merged_groups
    
    def _create_context_group(
        self, 
        chunks: List[ContextualChunk], 
        document: Document, 
        group_type: ContextGroupType,
        group_metadata: Optional[Dict[str, Any]] = None
    ) -> ContextGroup:
        """Create a context group from chunks"""
        
        # Generate group ID
        group_id = f"{document.metadata.title}_{group_type.value}_{len(chunks)}chunks"
        group_id = re.sub(r'[^\w\-_]', '_', group_id)  # Clean ID
        
        # Calculate page range
        start_page = None
        end_page = None
        
        if chunks:
            pages = []
            for chunk in chunks:
                if chunk.page_range:
                    pages.extend([chunk.page_range[0], chunk.page_range[1]])
            
            if pages:
                start_page = min(pages)
                end_page = max(pages)
        
        # Create structure info
        structure_info = StructureInfo(
            document_type=document.metadata.document_type,
            total_pages=document.metadata.page_count,
            total_segments=sum(len(chunk.segment_references) for chunk in chunks),
            has_tables=any(chunk.chunk_type.value == "table_data" for chunk in chunks),
            has_images=any(chunk.visual_elements for chunk in chunks),
            has_charts=any(
                any(ve.element_type.value in ["chart", "graph"] for ve in chunk.visual_elements)
                for chunk in chunks
            )
        )
        
        # Create group
        group = ContextGroup(
            group_id=group_id,
            document_id=document.metadata.title or "unknown",
            group_type=group_type,
            chunks=chunks,
            structure_info=structure_info,
            start_page=start_page,
            end_page=end_page,
            group_metadata=group_metadata or {}
        )
        
        return group
    
    def _update_chunk_positions(self, groups: List[ContextGroup]):
        """Update chunk positions within groups"""
        for group in groups:
            for i, chunk in enumerate(group.chunks):
                from .chunk_models import ChunkPosition
                chunk.chunk_position = ChunkPosition(
                    group_id=group.group_id,
                    position=i,
                    total_chunks=len(group.chunks)
                )
    
    def _create_fallback_group(self, chunks: List[ContextualChunk], document: Document) -> List[ContextGroup]:
        """Create fallback single group when grouping fails"""
        if not chunks:
            return []
        
        group = self._create_context_group(
            chunks=chunks,
            document=document,
            group_type=ContextGroupType.PAGE_RANGE,
            group_metadata={"fallback": True}
        )
        
        self._update_chunk_positions([group])
        
        return [group]