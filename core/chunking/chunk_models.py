"""Enhanced data models for contextual chunking"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from core.parsers import DocumentType, VisualElement


class ChunkType(Enum):
    """Type of chunk content"""
    CONTENT = "content"                  # Main textual content
    VISUAL_CONTEXT = "visual_context"   # Visual element descriptions  
    METADATA = "metadata"               # Document/section metadata
    HYBRID = "hybrid"                   # Mixed content + visual
    TABLE_DATA = "table_data"           # Structured table content
    CODE_BLOCK = "code_block"           # Code or formula content


class ContextGroupType(Enum):
    """Type of context group"""
    SECTION = "section"                 # Document section/chapter
    TOPIC = "topic"                     # Semantic topic boundary
    VISUAL_UNIT = "visual_unit"         # Chunks sharing visual elements
    PAGE_RANGE = "page_range"           # Page-based grouping
    SLIDE_SET = "slide_set"             # Presentation slide grouping
    SHEET_GROUP = "sheet_group"         # Spreadsheet sheet grouping
    HEADING_HIERARCHY = "heading_hierarchy"  # Heading-based grouping


class ChunkingStrategy(Enum):
    """Chunking strategy type"""
    STRUCTURE_AWARE = "structure_aware"     # Use document structure
    FIXED_SIZE = "fixed_size"               # Fixed token count
    SEMANTIC = "semantic"                   # Semantic boundaries
    VISUAL_COHERENCE = "visual_coherence"   # Visual element coherence
    HYBRID = "hybrid"                       # Combined approach


class OverlapStrategy(Enum):
    """Strategy for chunk overlapping"""
    NONE = "none"                       # No overlap
    SENTENCE_OVERLAP = "sentence"       # Complete sentences
    PARAGRAPH_OVERLAP = "paragraph"     # Whole paragraphs  
    SEMANTIC_OVERLAP = "semantic"       # Semantically relevant content
    VISUAL_OVERLAP = "visual"           # Visual elements in multiple chunks
    TOKEN_OVERLAP = "token"             # Fixed token count overlap


@dataclass
class ChunkPosition:
    """Position information within a context group"""
    group_id: str
    position: int                       # 0-based position in group
    total_chunks: int                   # Total chunks in group
    
    @property
    def is_first(self) -> bool:
        """Check if this is the first chunk in group"""
        return self.position == 0
    
    @property
    def is_last(self) -> bool:
        """Check if this is the last chunk in group"""
        return self.position == self.total_chunks - 1
    
    @property
    def is_middle(self) -> bool:
        """Check if this is a middle chunk"""
        return not (self.is_first or self.is_last)
    
    @property
    def progress_ratio(self) -> float:
        """Progress through the group (0.0 to 1.0)"""
        if self.total_chunks <= 1:
            return 1.0
        return self.position / (self.total_chunks - 1)


@dataclass
class ContextInheritance:
    """Context inheritance metadata"""
    source_chunk_id: str                    # Source chunk that generated context
    context_chain: List[str]                # Full chain of context inheritance
    context_freshness: int                  # How many chunks since context creation
    context_relevance_score: float         # Relevance of inherited context (0-1)
    generation_timestamp: datetime         # When context was generated
    context_summary_tokens: int            # Token count of context summary
    
    @property
    def is_fresh(self) -> bool:
        """Check if context is still fresh (freshness < 3)"""
        return self.context_freshness < 3
    
    @property
    def is_relevant(self) -> bool:
        """Check if context is still relevant (score > 0.5)"""
        return self.context_relevance_score > 0.5


@dataclass
class StructureInfo:
    """Document structure information for grouping"""
    document_type: DocumentType
    total_pages: Optional[int] = None
    total_segments: int = 0
    heading_levels: List[str] = field(default_factory=list)
    has_tables: bool = False
    has_images: bool = False
    has_charts: bool = False
    section_markers: List[str] = field(default_factory=list)
    custom_structure: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingStats:
    """Statistics about the chunking process"""
    total_chunks: int = 0
    total_context_groups: int = 0
    avg_chunk_tokens: float = 0.0
    avg_context_tokens: float = 0.0
    processing_time_seconds: float = 0.0
    context_generation_time: float = 0.0
    
    # Quality metrics
    chunks_with_context: int = 0
    chunks_with_visuals: int = 0
    empty_chunks: int = 0
    oversized_chunks: int = 0
    
    # Token distribution
    total_content_tokens: int = 0
    total_context_tokens: int = 0
    total_visual_tokens: int = 0
    
    @property
    def context_coverage_ratio(self) -> float:
        """Ratio of chunks that have inherited context"""
        if self.total_chunks == 0:
            return 0.0
        return self.chunks_with_context / self.total_chunks
    
    @property
    def visual_coverage_ratio(self) -> float:
        """Ratio of chunks that contain visual elements"""
        if self.total_chunks == 0:
            return 0.0
        return self.chunks_with_visuals / self.total_chunks
    
    @property
    def efficiency_ratio(self) -> float:
        """Ratio of content tokens to total tokens"""
        total_tokens = self.total_content_tokens + self.total_context_tokens + self.total_visual_tokens
        if total_tokens == 0:
            return 0.0
        return self.total_content_tokens / total_tokens


@dataclass
class ContextualChunk:
    """Enhanced chunk with context inheritance capabilities"""
    
    # Core identification
    chunk_id: str
    source_document_id: str
    
    # Content
    content: str
    token_count: int
    chunk_type: ChunkType
    
    # Source references
    segment_references: List[str] = field(default_factory=list)
    page_range: Optional[tuple[int, int]] = None
    visual_elements: List[VisualElement] = field(default_factory=list)
    
    # Context inheritance
    context_group_id: Optional[str] = None
    chunk_position: Optional[ChunkPosition] = None
    inherited_context: Optional[str] = None
    generates_context: bool = False
    context_inheritance: Optional[ContextInheritance] = None
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.now)
    chunking_strategy: Optional[ChunkingStrategy] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chunk data after initialization"""
        if not self.content.strip():
            raise ValueError(f"Chunk {self.chunk_id} cannot have empty content")
        
        if self.token_count <= 0:
            raise ValueError(f"Chunk {self.chunk_id} must have positive token count")
    
    @classmethod
    def create_chunk_id(cls, document_id: str, group_id: str, position: int) -> str:
        """Create a unique chunk ID"""
        return f"{document_id}__group_{group_id}__chunk_{position:03d}"
    
    def get_enhanced_content(self) -> str:
        """Get content with visual descriptions integrated"""
        content_parts = [self.content]
        
        # Add visual element descriptions
        for visual in self.visual_elements:
            if visual.vlm_description:
                visual_context = f"\n[VISUAL: {visual.element_type.value.upper()}]\n{visual.vlm_description}"
                content_parts.append(visual_context)
            
            # Add structured data if available
            if visual.extracted_data:
                data_context = f"\n[DATA: {self._format_extracted_data(visual.extracted_data)}]"
                content_parts.append(data_context)
        
        return "\n".join(content_parts)
    
    def get_context_prompt(self, task_template: str) -> str:
        """Generate LLM prompt with inherited context"""
        prompt_parts = []
        
        # Add inherited context if available
        if self.inherited_context:
            prompt_parts.append(f"VORHERIGER KONTEXT:\n{self.inherited_context}\n")
        
        # Add main task
        prompt_parts.append(f"AUFGABE:\n{task_template}\n")
        
        # Add enhanced content
        prompt_parts.append(f"TEXT:\n{self.get_enhanced_content()}")
        
        return "\n".join(prompt_parts)
    
    def get_context_generation_prompt(self, task_template: str) -> str:
        """Generate prompt for context summary creation (first chunk in group)"""
        prompt_parts = [
            f"HAUPTAUFGABE: {task_template}",
            "",
            "ZUSÄTZLICHE AUFGABE: Erstelle eine Kontextzusammenfassung für nachfolgende Textabschnitte.",
            "",
            "Die Kontextzusammenfassung soll:",
            "1. Kernthemen und wichtige Konzepte erfassen",
            "2. Relevante Definitionen und Erklärungen enthalten",
            "3. Wichtige Referenzen und Verweise festhalten", 
            "4. Den thematischen 'roten Faden' bewahren",
            "5. Maximal 300 Tokens umfassen",
            "",
            f"TEXT:\n{self.get_enhanced_content()}",
            "",
            "AUSGABEFORMAT:",
            "HAUPTAUFGABE ERGEBNIS:",
            "[Hier das Ergebnis der Hauptaufgabe]",
            "",
            "KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:",
            "[Kompakte Zusammenfassung der relevanten Kontextinformationen]"
        ]
        
        return "\n".join(prompt_parts)
    
    def _format_extracted_data(self, data: Dict[str, Any]) -> str:
        """Format extracted data for display"""
        if not data:
            return ""
        
        # Handle different data types
        if "chart_type" in data:
            return f"Chart ({data['chart_type']}): {data.get('data_points', {})}"
        elif "table_data" in data:
            return f"Table: {data['table_data']}"
        else:
            # Generic formatting
            return str(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "content": self.content,
            "token_count": self.token_count,
            "chunk_type": self.chunk_type.value,
            "segment_references": self.segment_references,
            "page_range": self.page_range,
            "visual_element_count": len(self.visual_elements),
            "context_group_id": self.context_group_id,
            "has_inherited_context": self.inherited_context is not None,
            "generates_context": self.generates_context,
            "chunk_position": {
                "group_id": self.chunk_position.group_id,
                "position": self.chunk_position.position,
                "total_chunks": self.chunk_position.total_chunks,
                "is_first": self.chunk_position.is_first,
                "is_last": self.chunk_position.is_last
            } if self.chunk_position else None,
            "created_at": self.created_at.isoformat(),
            "chunking_strategy": self.chunking_strategy.value if self.chunking_strategy else None,
            "processing_metadata": self.processing_metadata
        }


@dataclass
class ContextGroup:
    """Group of chunks sharing inherited context"""
    
    group_id: str
    document_id: str
    group_type: ContextGroupType
    
    # Content
    chunks: List[ContextualChunk] = field(default_factory=list)
    context_summary: Optional[str] = None
    
    # Structure information
    structure_info: Optional[StructureInfo] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.now)
    context_generated_at: Optional[datetime] = None
    group_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks in this group"""
        return len(self.chunks)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens across all chunks in group"""
        return sum(chunk.token_count for chunk in self.chunks)
    
    @property
    def has_context(self) -> bool:
        """Check if group has generated context summary"""
        return self.context_summary is not None
    
    @property
    def visual_element_count(self) -> int:
        """Total visual elements across all chunks"""
        return sum(len(chunk.visual_elements) for chunk in self.chunks)
    
    def get_first_chunk(self) -> Optional[ContextualChunk]:
        """Get the first chunk in the group"""
        if not self.chunks:
            return None
        return min(self.chunks, key=lambda c: c.chunk_position.position if c.chunk_position else 0)
    
    def get_last_chunk(self) -> Optional[ContextualChunk]:
        """Get the last chunk in the group"""
        if not self.chunks:
            return None
        return max(self.chunks, key=lambda c: c.chunk_position.position if c.chunk_position else 0)
    
    def add_chunk(self, chunk: ContextualChunk):
        """Add a chunk to this group"""
        chunk.context_group_id = self.group_id
        self.chunks.append(chunk)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary for serialization"""
        return {
            "group_id": self.group_id,
            "document_id": self.document_id,
            "group_type": self.group_type.value,
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
            "has_context": self.has_context,
            "visual_element_count": self.visual_element_count,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "created_at": self.created_at.isoformat(),
            "context_generated_at": self.context_generated_at.isoformat() if self.context_generated_at else None,
            "group_metadata": self.group_metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


@dataclass
class ChunkingResult:
    """Complete result of the chunking process"""
    
    # Identification
    document_id: str
    source_document: Any  # Document object from parsers
    
    # Results
    contextual_chunks: List[ContextualChunk] = field(default_factory=list)
    context_groups: List[ContextGroup] = field(default_factory=list)
    
    # Processing information
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_AWARE
    processing_stats: Optional[ChunkingStats] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    processing_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_chunks(self) -> int:
        """Total number of chunks"""
        return len(self.contextual_chunks)
    
    @property
    def total_groups(self) -> int:
        """Total number of context groups"""
        return len(self.context_groups)
    
    @property
    def chunks_with_context(self) -> int:
        """Number of chunks with inherited context"""
        return sum(1 for chunk in self.contextual_chunks if chunk.inherited_context)
    
    @property
    def success_rate(self) -> float:
        """Rate of successful chunk creation"""
        if not self.processing_stats:
            return 1.0
        total_attempted = self.processing_stats.total_chunks + self.processing_stats.empty_chunks
        if total_attempted == 0:
            return 1.0
        return self.processing_stats.total_chunks / total_attempted
    
    def get_chunks_by_group(self, group_id: str) -> List[ContextualChunk]:
        """Get all chunks belonging to a specific group"""
        return [chunk for chunk in self.contextual_chunks if chunk.context_group_id == group_id]
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[ContextualChunk]:
        """Get all chunks of a specific type"""
        return [chunk for chunk in self.contextual_chunks if chunk.chunk_type == chunk_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "total_groups": self.total_groups,
            "chunks_with_context": self.chunks_with_context,
            "success_rate": self.success_rate,
            "chunking_strategy": self.chunking_strategy.value,
            "created_at": self.created_at.isoformat(),
            "processing_config": self.processing_config,
            "processing_stats": self.processing_stats.__dict__ if self.processing_stats else None,
            "context_groups": [group.to_dict() for group in self.context_groups],
            "contextual_chunks": [chunk.to_dict() for chunk in self.contextual_chunks]
        }