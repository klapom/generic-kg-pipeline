"""Base chunker class with common functionality"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from plugins.parsers.base_parser import Document, Segment, VisualElement
from .chunk_models import (
    ContextualChunk,
    ChunkType,
    ChunkingStrategy,
    ChunkingStats,
    StructureInfo
)

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for document chunkers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base chunker with configuration"""
        self.config = config
        self.chunking_config = config.get("chunking", {})
        self.strategy = ChunkingStrategy(config.get("default_strategy", "structure_aware"))
        
        # Token counting configuration
        self.token_estimation_model = self.chunking_config.get("token_management", {}).get("estimation_model", "gpt-3.5-turbo")
        self.buffer_ratio = self.chunking_config.get("token_management", {}).get("buffer_ratio", 0.1)
        
        logger.info(f"Initialized {self.__class__.__name__} with strategy: {self.strategy.value}")
    
    @abstractmethod
    def chunk_document(self, document: Document, strategy_config: Dict[str, Any]) -> List[ContextualChunk]:
        """Abstract method to chunk a document"""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.token_estimation_model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to simple approximation
            return len(text.split()) * 1.3  # Rough approximation
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}, using fallback")
            return len(text.split()) * 1.3
    
    def calculate_token_budget(self, max_tokens: int, context_tokens: int = 0) -> int:
        """Calculate available tokens for content after accounting for context"""
        # Apply buffer ratio
        effective_max = int(max_tokens * (1 - self.buffer_ratio))
        
        # Subtract context tokens
        content_budget = effective_max - context_tokens
        
        # Ensure minimum viable content space
        min_content = max_tokens * 0.3  # At least 30% for content
        content_budget = max(content_budget, min_content)
        
        return int(content_budget)
    
    def extract_structure_info(self, document: Document) -> StructureInfo:
        """Extract structure information from document"""
        structure_info = StructureInfo(
            document_type=document.metadata.document_type,
            total_pages=document.metadata.page_count,
            total_segments=len(document.segments),
            has_tables=any(seg.segment_type == "table" for seg in document.segments),
            has_images=len(document.visual_elements) > 0,
            has_charts=any(ve.element_type.value in ["chart", "graph"] for ve in document.visual_elements)
        )
        
        # Extract heading levels
        headings = []
        for segment in document.segments:
            if segment.segment_type in ["heading", "title"]:
                headings.append(segment.content)
        structure_info.heading_levels = headings
        
        # Extract section markers
        section_markers = []
        for segment in document.segments:
            if self._is_section_marker(segment):
                section_markers.append(segment.content)
        structure_info.section_markers = section_markers
        
        return structure_info
    
    def _is_section_marker(self, segment: Segment) -> bool:
        """Check if segment is a section marker"""
        if segment.segment_type in ["heading", "title"]:
            return True
        
        # Check for pattern-based section markers
        content = segment.content.strip()
        
        # Numbered sections: "1. Introduction", "2.1 Methods"
        if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', content):
            return True
        
        # All caps short text (likely headings)
        if content.isupper() and len(content) < 100:
            return True
        
        return False
    
    def segments_to_chunks(
        self, 
        segments: List[Segment], 
        document_id: str,
        strategy_config: Dict[str, Any]
    ) -> List[ContextualChunk]:
        """Convert segments to contextual chunks using structure-aware strategy"""
        chunks = []
        current_chunk_content = []
        current_chunk_segments = []
        current_chunk_visuals = []
        current_token_count = 0
        
        max_tokens = strategy_config.get("max_tokens", 2000)
        min_tokens = strategy_config.get("min_tokens", 200)
        
        for segment in segments:
            # Estimate tokens for this segment
            segment_tokens = self.estimate_tokens(segment.content)
            
            # Get visual elements for this segment
            segment_visuals = self._get_segment_visuals(segment, document_id)
            visual_tokens = sum(self.estimate_tokens(ve.vlm_description or "") for ve in segment_visuals)
            
            # Calculate projected token count
            projected_tokens = current_token_count + segment_tokens + visual_tokens
            
            # Check if we need to create a new chunk
            if projected_tokens > max_tokens and current_chunk_content:
                # Create chunk from current content
                chunk = self._create_chunk_from_content(
                    content=current_chunk_content,
                    segments=current_chunk_segments,
                    visuals=current_chunk_visuals,
                    document_id=document_id,
                    chunk_index=len(chunks),
                    strategy_config=strategy_config
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_content = [segment.content]
                current_chunk_segments = [segment]
                current_chunk_visuals = segment_visuals
                current_token_count = segment_tokens + visual_tokens
            else:
                # Add to current chunk
                current_chunk_content.append(segment.content)
                current_chunk_segments.append(segment)
                current_chunk_visuals.extend(segment_visuals)
                current_token_count = projected_tokens
        
        # Create final chunk
        if current_chunk_content:
            chunk = self._create_chunk_from_content(
                content=current_chunk_content,
                segments=current_chunk_segments,
                visuals=current_chunk_visuals,
                document_id=document_id,
                chunk_index=len(chunks),
                strategy_config=strategy_config
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_segment_visuals(self, segment: Segment, document_id: str) -> List[VisualElement]:
        """Get visual elements associated with a segment"""
        # This would need to be implemented based on the document structure
        # For now, return empty list as placeholder
        return []
    
    def _create_chunk_from_content(
        self,
        content: List[str],
        segments: List[Segment],
        visuals: List[VisualElement],
        document_id: str,
        chunk_index: int,
        strategy_config: Dict[str, Any]
    ) -> ContextualChunk:
        """Create a contextual chunk from content and metadata"""
        
        # Join content
        chunk_content = "\n\n".join(content)
        
        # Estimate tokens
        token_count = self.estimate_tokens(chunk_content)
        
        # Add visual descriptions
        if visuals:
            visual_content = []
            for visual in visuals:
                if visual.vlm_description:
                    visual_content.append(f"[VISUAL: {visual.vlm_description}]")
                if visual.extracted_data:
                    visual_content.append(f"[DATA: {visual.extracted_data}]")
            
            if visual_content:
                chunk_content += "\n\n" + "\n".join(visual_content)
                token_count = self.estimate_tokens(chunk_content)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(segments, visuals)
        
        # Create chunk ID
        chunk_id = f"{document_id}__chunk_{chunk_index:03d}"
        
        # Extract segment references
        segment_refs = [f"segment_{seg.segment_index}" for seg in segments]
        
        # Calculate page range
        page_range = self._calculate_page_range(segments)
        
        # Create chunk
        chunk = ContextualChunk(
            chunk_id=chunk_id,
            source_document_id=document_id,
            content=chunk_content,
            token_count=token_count,
            chunk_type=chunk_type,
            segment_references=segment_refs,
            page_range=page_range,
            visual_elements=visuals,
            chunking_strategy=self.strategy,
            processing_metadata={
                "strategy_config": strategy_config,
                "source_segments": len(segments),
                "original_content_tokens": self.estimate_tokens("\n\n".join(content)),
                "visual_tokens": token_count - self.estimate_tokens("\n\n".join(content))
            }
        )
        
        return chunk
    
    def _determine_chunk_type(self, segments: List[Segment], visuals: List[VisualElement]) -> ChunkType:
        """Determine the type of chunk based on content"""
        if not segments:
            return ChunkType.METADATA
        
        # Check for tables
        if any(seg.segment_type == "table" for seg in segments):
            return ChunkType.TABLE_DATA
        
        # Check for mixed content with visuals
        if visuals and any(seg.segment_type in ["text", "paragraph"] for seg in segments):
            return ChunkType.HYBRID
        
        # Check for primarily visual content
        if visuals and not any(seg.segment_type in ["text", "paragraph"] for seg in segments):
            return ChunkType.VISUAL_CONTEXT
        
        # Default to content
        return ChunkType.CONTENT
    
    def _calculate_page_range(self, segments: List[Segment]) -> Optional[Tuple[int, int]]:
        """Calculate page range from segments"""
        if not segments:
            return None
        
        pages = [seg.page_number for seg in segments if seg.page_number is not None]
        if not pages:
            return None
        
        return (min(pages), max(pages))
    
    def create_chunking_stats(
        self, 
        chunks: List[ContextualChunk], 
        processing_time: float,
        context_generation_time: float = 0.0
    ) -> ChunkingStats:
        """Create statistics for chunking process"""
        
        stats = ChunkingStats(
            total_chunks=len(chunks),
            avg_chunk_tokens=sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0,
            processing_time_seconds=processing_time,
            context_generation_time=context_generation_time
        )
        
        # Calculate detailed statistics
        stats.chunks_with_context = sum(1 for chunk in chunks if chunk.inherited_context)
        stats.chunks_with_visuals = sum(1 for chunk in chunks if chunk.visual_elements)
        stats.empty_chunks = sum(1 for chunk in chunks if not chunk.content.strip())
        
        # Token statistics
        stats.total_content_tokens = sum(chunk.token_count for chunk in chunks)
        stats.total_context_tokens = sum(
            self.estimate_tokens(chunk.inherited_context or "") for chunk in chunks
        )
        
        # Visual tokens
        stats.total_visual_tokens = sum(
            sum(self.estimate_tokens(ve.vlm_description or "") for ve in chunk.visual_elements)
            for chunk in chunks
        )
        
        return stats
    
    def validate_chunks(self, chunks: List[ContextualChunk], strategy_config: Dict[str, Any]) -> List[ContextualChunk]:
        """Validate and filter chunks based on quality criteria"""
        valid_chunks = []
        
        max_tokens = strategy_config.get("max_tokens", 2000)
        min_tokens = strategy_config.get("min_tokens", 200)
        hard_limit = int(max_tokens * 1.2)  # 20% over limit
        
        for chunk in chunks:
            # Check token limits
            if chunk.token_count > hard_limit:
                logger.warning(f"Chunk {chunk.chunk_id} exceeds hard limit ({chunk.token_count} > {hard_limit})")
                continue
            
            # Check minimum content
            if chunk.token_count < min_tokens:
                logger.warning(f"Chunk {chunk.chunk_id} below minimum tokens ({chunk.token_count} < {min_tokens})")
                continue
            
            # Check content quality
            if not chunk.content.strip():
                logger.warning(f"Chunk {chunk.chunk_id} has empty content")
                continue
            
            valid_chunks.append(chunk)
        
        logger.info(f"Validated {len(valid_chunks)} out of {len(chunks)} chunks")
        return valid_chunks