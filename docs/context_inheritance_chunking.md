# Context Inheritance Chunking Strategy

## Konzept: Intelligente Kontext-Vererbung

### Problem mit traditionellem Overlapping:
```
Chunk 1: [A B C D E] F G H I J
Chunk 2: [H I J] K L M N O P Q
Chunk 3: [N O P Q] R S T U V W
```
**Nachteile:**
- Redundante Information wird wiederholt verarbeitet
- Kein semantischer Kontext über Chunk-Grenzen hinweg
- LLM verliert den "roten Faden" bei komplexen Dokumenten

### Lösung: Context Inheritance
```
Chunk 1: [A B C D E F G] → Generiert: CONTEXT_SUMMARY_1
Chunk 2: CONTEXT_SUMMARY_1 + [H I J K L M] → Generiert: CONTEXT_SUMMARY_2  
Chunk 3: CONTEXT_SUMMARY_2 + [N O P Q R S] → Generiert: CONTEXT_SUMMARY_3
```

## Erweiterte Chunk-Datenstruktur

```python
@dataclass
class ContextualChunk(Chunk):
    """Enhanced Chunk with Context Inheritance"""
    
    # Existing fields from base Chunk class
    content: str
    chunk_id: str
    token_count: int
    
    # NEW: Context Inheritance fields
    context_summary: Optional[str] = None           # Inherited context from previous chunks
    generates_context: bool = False                 # Should this chunk generate context summary?
    context_group_id: str = None                   # Groups related chunks together
    chunk_position: ChunkPosition = None           # Position within context group
    
    # Context metadata
    context_inheritance: Optional[ContextInheritance] = None
    
    def get_llm_prompt(self, task_template: str) -> str:
        """Generate LLM prompt with inherited context"""
        
    def get_context_generation_prompt(self) -> str:
        """Generate prompt for context summary creation"""

@dataclass
class ContextInheritance:
    """Context inheritance metadata"""
    parent_chunk_id: Optional[str]                 # Source of inherited context
    context_chain: List[str]                       # Full chain of context inheritance
    context_freshness: int                         # How many chunks since context creation
    context_relevance_score: float                # Relevance of inherited context
    
@dataclass 
class ChunkPosition:
    """Position within a context group"""
    group_id: str
    position: int                                  # 0-based position in group
    total_chunks: int                             # Total chunks in group
    is_first: bool                                # First chunk in group
    is_last: bool                                 # Last chunk in group
    
    @property
    def is_middle(self) -> bool:
        return not (self.is_first or self.is_last)
```

## Context-Group Strategie

```python
class ContextGrouper:
    """Groups chunks into logical context units"""
    
    def __init__(self, config: ContextGroupingConfig):
        self.config = config
        
    def group_chunks(self, chunks: List[Chunk], document: Document) -> List[ContextGroup]:
        """
        Group chunks into logical context units based on:
        1. Document structure (chapters, sections)
        2. Topic boundaries (semantic similarity)
        3. Visual element boundaries
        4. Page/slide boundaries
        """
        groups = []
        
        # Strategy 1: Structure-based grouping
        if document.metadata.document_type == DocumentType.PDF:
            groups = self._group_by_pdf_structure(chunks, document)
        elif document.metadata.document_type == DocumentType.PPTX:
            groups = self._group_by_slides(chunks, document)
        elif document.metadata.document_type in [DocumentType.DOCX, DocumentType.XLSX]:
            groups = self._group_by_sections(chunks, document)
        
        # Strategy 2: Semantic boundary detection
        if self.config.use_semantic_boundaries:
            groups = self._refine_with_semantic_boundaries(groups, chunks)
            
        # Strategy 3: Visual element coherence
        if self.config.respect_visual_boundaries:
            groups = self._respect_visual_coherence(groups, chunks, document)
            
        return groups
    
    def _group_by_pdf_structure(self, chunks: List[Chunk], document: Document) -> List[ContextGroup]:
        """Group PDF chunks by structural elements"""
        groups = []
        current_group = []
        current_section = None
        
        for chunk in chunks:
            # Detect section changes (headings, chapter markers)
            chunk_sections = self._detect_sections_in_chunk(chunk)
            
            if chunk_sections and chunk_sections != current_section:
                # Start new group at section boundary
                if current_group:
                    groups.append(ContextGroup(
                        group_id=f"section_{current_section}",
                        chunks=current_group,
                        context_type=ContextType.SECTION,
                        metadata={"section": current_section}
                    ))
                
                current_group = [chunk]
                current_section = chunk_sections
            else:
                current_group.append(chunk)
        
        # Add final group
        if current_group:
            groups.append(ContextGroup(
                group_id=f"section_{current_section}",
                chunks=current_group,
                context_type=ContextType.SECTION
            ))
            
        return groups
    
    def _detect_sections_in_chunk(self, chunk: Chunk) -> Optional[str]:
        """Detect section markers in chunk content"""
        # Look for heading patterns
        lines = chunk.content.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            
            # Pattern 1: Numbered sections (1. Introduction, 2.1 Methods)
            if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', line):
                return line
            
            # Pattern 2: All caps headings
            if line.isupper() and len(line) < 100:
                return line
                
            # Pattern 3: Segment type is heading
            if any(seg.segment_type == "heading" for seg in chunk.segment_references):
                return line
                
        return None

@dataclass
class ContextGroup:
    """Group of related chunks sharing context"""
    group_id: str
    chunks: List[Chunk]
    context_type: ContextType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class ContextType(Enum):
    SECTION = "section"          # Document section/chapter
    TOPIC = "topic"              # Semantic topic boundary  
    VISUAL_UNIT = "visual_unit"  # Chunks sharing visual elements
    PAGE_RANGE = "page_range"    # Page-based grouping
    SLIDE_SET = "slide_set"      # Presentation slide grouping
```

## Context Summary Generation

```python
class ContextSummaryGenerator:
    """Generates context summaries for chunk inheritance"""
    
    def __init__(self, llm_client, config: ContextSummaryConfig):
        self.llm_client = llm_client
        self.config = config
        
    async def generate_context_summary(
        self, 
        chunk: ContextualChunk, 
        task_context: str,
        previous_context: Optional[str] = None
    ) -> str:
        """
        Generate context summary for inheritance to next chunks
        
        Args:
            chunk: The chunk to summarize
            task_context: What task will be performed on following chunks
            previous_context: Any inherited context from previous chunks
        """
        
        # Build context generation prompt
        prompt = self._build_context_prompt(chunk, task_context, previous_context)
        
        # Generate summary
        response = await self.llm_client.generate_completion(
            prompt=prompt,
            max_tokens=self.config.max_context_tokens,
            temperature=0.1  # Low temperature for consistent summaries
        )
        
        context_summary = self._extract_context_summary(response)
        
        # Validate and clean summary
        context_summary = self._validate_context_summary(context_summary, chunk)
        
        return context_summary
    
    def _build_context_prompt(
        self, 
        chunk: ContextualChunk, 
        task_context: str,
        previous_context: Optional[str]
    ) -> str:
        """Build prompt for context summary generation"""
        
        base_prompt = f"""
Du analysierst den folgenden Textabschnitt und führst dabei zwei Aufgaben aus:

HAUPTAUFGABE: {task_context}

ZUSÄTZLICHE AUFGABE: Erstelle eine prägnante Zusammenfassung des Kontexts, der für nachfolgende Textabschnitte relevant ist.

Die Kontextzusammenfassung soll:
1. Kernthemen und wichtige Konzepte erfassen
2. Relevante Definitionen und Erklärungen enthalten  
3. Wichtige Verweise und Referenzen festhalten
4. Den "roten Faden" für nachfolgende Abschnitte bewahren
5. Maximal {self.config.max_context_tokens} Tokens umfassen

TEXT:
{chunk.content}
"""

        if previous_context:
            base_prompt = f"""
VORHERIGER KONTEXT (von vorherigen Abschnitten):
{previous_context}

{base_prompt}

WICHTIG: Aktualisiere und erweitere den vorherigen Kontext mit neuen Informationen aus dem aktuellen Textabschnitt.
"""

        if chunk.visual_elements:
            visual_context = self._format_visual_context(chunk.visual_elements)
            base_prompt += f"""

VISUELLE ELEMENTE:
{visual_context}

Berücksichtige diese visuellen Elemente in der Kontextzusammenfassung.
"""

        base_prompt += """

AUSGABEFORMAT:
1. [Hauptaufgabe Ergebnis]

2. KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:
[Prägnante Zusammenfassung der relevanten Kontextinformationen]
"""

        return base_prompt
    
    def _extract_context_summary(self, llm_response: str) -> str:
        """Extract context summary from LLM response"""
        # Look for context section marker
        context_markers = [
            "KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:",
            "CONTEXT FOR FOLLOWING SECTIONS:", 
            "CONTEXT SUMMARY:",
            "ZUSAMMENFASSUNG:",
        ]
        
        for marker in context_markers:
            if marker in llm_response:
                parts = llm_response.split(marker, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # Fallback: use last paragraph if no marker found
        paragraphs = llm_response.split('\n\n')
        return paragraphs[-1].strip() if paragraphs else ""
    
    def _validate_context_summary(self, summary: str, chunk: ContextualChunk) -> str:
        """Validate and clean context summary"""
        # Remove empty lines and excessive whitespace
        summary = re.sub(r'\n\s*\n', '\n', summary).strip()
        
        # Ensure reasonable length
        if len(summary) > self.config.max_context_length:
            # Truncate at sentence boundary
            sentences = summary.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > self.config.max_context_length:
                    break
                truncated.append(sentence)
                current_length += len(sentence)
            
            summary = '. '.join(truncated)
            if not summary.endswith('.'):
                summary += '.'
        
        return summary
```

## Integration in den Chunking-Workflow

```python
class ContextInheritanceChunker(ContentChunker):
    """Enhanced chunker with context inheritance"""
    
    async def chunk_document_with_context_inheritance(
        self, 
        document: Document,
        task_template: str
    ) -> List[ContextualChunk]:
        """
        Main chunking method with context inheritance
        
        Workflow:
        1. Create base chunks using structure-aware strategy
        2. Group chunks into logical context units
        3. Process each group with context inheritance
        4. Generate context summaries for first chunks
        5. Inherit and propagate context through group
        """
        
        # Step 1: Create base chunks
        base_chunks = self.create_structure_aware_chunks(document)
        
        # Step 2: Group chunks into context units
        context_groups = self.context_grouper.group_chunks(base_chunks, document)
        
        # Step 3: Process each group with context inheritance
        all_contextual_chunks = []
        
        for group in context_groups:
            contextual_chunks = await self._process_context_group(group, task_template)
            all_contextual_chunks.extend(contextual_chunks)
        
        return all_contextual_chunks
    
    async def _process_context_group(
        self, 
        group: ContextGroup, 
        task_template: str
    ) -> List[ContextualChunk]:
        """Process a context group with inheritance"""
        
        contextual_chunks = []
        current_context = None
        
        for i, chunk in enumerate(group.chunks):
            # Create contextual chunk
            contextual_chunk = ContextualChunk(
                **chunk.__dict__,  # Copy base chunk properties
                context_group_id=group.group_id,
                chunk_position=ChunkPosition(
                    group_id=group.group_id,
                    position=i,
                    total_chunks=len(group.chunks),
                    is_first=(i == 0),
                    is_last=(i == len(group.chunks) - 1)
                ),
                context_summary=current_context,
                generates_context=(i == 0)  # First chunk generates context
            )
            
            # Generate context summary for first chunk
            if contextual_chunk.generates_context:
                current_context = await self.context_summary_generator.generate_context_summary(
                    chunk=contextual_chunk,
                    task_context=task_template,
                    previous_context=None
                )
                
                # Update context for next chunks
                contextual_chunk.context_summary = current_context
            
            # Add inheritance metadata
            if i > 0:
                contextual_chunk.context_inheritance = ContextInheritance(
                    parent_chunk_id=contextual_chunks[0].chunk_id,  # First chunk is context source
                    context_chain=[contextual_chunks[0].chunk_id],
                    context_freshness=i,
                    context_relevance_score=1.0 - (i * 0.1)  # Decay over distance
                )
            
            contextual_chunks.append(contextual_chunk)
        
        return contextual_chunks
```

## Konfiguration

```yaml
# config/context_inheritance.yaml
context_inheritance:
  enabled: true
  
  grouping:
    strategy: "hybrid"  # structure, semantic, hybrid
    max_group_size: 8   # Maximum chunks per context group
    min_group_size: 2   # Minimum chunks to enable inheritance
    respect_visual_boundaries: true
    use_semantic_boundaries: true
    
  context_summary:
    max_context_tokens: 300      # Max tokens for context summary
    max_context_length: 1500     # Max characters for context summary
    context_decay_rate: 0.1      # How much context relevance decays per chunk
    refresh_context_after: 5     # Regenerate context after N chunks
    
  prompt_templates:
    context_generation: "templates/context_generation.txt"
    task_with_context: "templates/task_with_context.txt"
```

## Vorteile dieser Strategie:

### 1. **Semantische Kohärenz**
- LLM behält den "roten Faden" über Chunk-Grenzen hinweg
- Besseres Verständnis von Referenzen und Zusammenhängen
- Reduzierte Fragmentierung von zusammenhängenden Konzepten

### 2. **Effizienz**
- Keine redundante Verarbeitung von überlappenden Inhalten
- Kompakte Kontextzusammenfassungen statt vollständiger Wiederholungen
- Intelligente Token-Nutzung

### 3. **Qualitätsverbesserung**
- Konsistentere Extraktion von Triples über Dokumentgrenzen hinweg
- Bessere Erkennung von Entitäten-Beziehungen
- Präzisere Attributzuordnungen

### 4. **Multi-Modal Integration**
- Visuelle Elemente werden in Kontextzusammenfassungen berücksichtigt
- Beschreibungen von Diagrammen/Charts propagieren zu verwandten Textabschnitten
- Bessere Verknüpfung von Text und visuellen Inhalten

Das ist definitiv **die bessere Strategie** gegenüber einfachem Overlapping! Soll ich das in den `ContentChunker` implementieren?