# Content Chunking mit Context Inheritance - Implementierungsplan

## 1. Analyse der bestehenden Architektur

### Vorhandene Strukturen (bereits implementiert):
```python
# plugins/parsers/base_parser.py
Document:
├── content: str                    # Volltext
├── segments: List[Segment]         # Strukturierte Abschnitte
├── visual_elements: List[VisualElement]  # Multi-modale Inhalte
├── metadata: DocumentMetadata      # Dokument-Metadaten
└── raw_data: Any                   # Original-Daten

Segment:
├── content: str                    # Text-Inhalt
├── segment_type: str               # "heading", "paragraph", "table", etc.
├── page_number: int                # Seitenzahl
├── segment_index: int              # Position im Dokument
├── visual_references: List[str]    # Verweise auf VisualElements
└── metadata: Dict                  # Segment-Metadaten
```

### Integration Points:
- ✅ **Parser Factory**: Liefert strukturierte Documents
- ✅ **LLM Clients**: Hochschul-LLM für Context Generation
- ✅ **Multi-Modal Content**: Visual Elements bereits verknüpft
- ❌ **Content Chunker**: Zu implementieren
- ❌ **Pipeline Integration**: Zu implementieren

## 2. Komponenten-Design

### 2.1 Core Chunking Components

```
core/
├── content_chunker.py              # Haupt-Chunker mit Context Inheritance
├── chunking/
│   ├── __init__.py
│   ├── base_chunker.py            # Abstract Base Chunker
│   ├── context_grouper.py         # Context Group Formation
│   ├── context_summarizer.py     # LLM-based Context Summary Generation
│   ├── chunk_models.py           # Enhanced Chunk Data Models
│   └── strategies/
│       ├── __init__.py
│       ├── structure_aware.py     # Structure-based Chunking
│       ├── semantic_chunking.py   # Semantic Boundary Detection
│       └── visual_coherence.py    # Visual Element Coherence
```

### 2.2 Data Models (Erweitert)

```python
# core/chunking/chunk_models.py

@dataclass
class ContextualChunk:
    """Enhanced Chunk with Context Inheritance capabilities"""
    # Base properties
    chunk_id: str
    content: str
    token_count: int
    
    # Source references
    source_document_id: str
    segment_references: List[str]
    visual_elements: List[VisualElement]
    
    # Context inheritance
    context_group_id: str
    chunk_position: ChunkPosition
    inherited_context: Optional[str] = None
    generates_context: bool = False
    
    # Metadata
    chunk_type: ChunkType
    processing_metadata: Dict[str, Any]
    
@dataclass
class ContextGroup:
    """Group of chunks sharing inherited context"""
    group_id: str
    document_id: str
    chunks: List[ContextualChunk]
    group_type: ContextGroupType
    structure_info: StructureInfo
    context_summary: Optional[str] = None
    
@dataclass
class ChunkingResult:
    """Complete chunking result with context inheritance"""
    document_id: str
    source_document: Document
    contextual_chunks: List[ContextualChunk]
    context_groups: List[ContextGroup]
    chunking_strategy: str
    processing_stats: ChunkingStats
```

### 2.3 Context Group Formation Strategy

```python
# core/chunking/context_grouper.py

class ContextGrouper:
    """Forms logical context groups from document segments"""
    
    def group_by_document_type(self, document: Document) -> List[ContextGroup]:
        """Main grouping dispatcher"""
        
    def _group_pdf_by_structure(self, document: Document) -> List[ContextGroup]:
        """PDF: Group by headings, sections, chapters using SmolDocling structure"""
        
    def _group_docx_by_headings(self, document: Document) -> List[ContextGroup]:
        """DOCX: Group by heading hierarchy and styles"""
        
    def _group_xlsx_by_sheets(self, document: Document) -> List[ContextGroup]:
        """XLSX: Group by sheets and logical data blocks"""
        
    def _group_pptx_by_topics(self, document: Document) -> List[ContextGroup]:
        """PPTX: Group slides by topic coherence"""
```

### 2.4 Context Summary Generation

```python
# core/chunking/context_summarizer.py

class ContextSummarizer:
    """Generates context summaries using LLM"""
    
    async def generate_context_summary(
        self,
        chunk: ContextualChunk,
        task_context: str,
        previous_context: Optional[str] = None
    ) -> ContextSummaryResult:
        """Generate context summary with dual-task prompting"""
        
    def _build_context_prompt(self, chunk, task_context, previous_context) -> str:
        """Build prompt for context generation"""
        
    def _extract_context_from_response(self, llm_response: str) -> str:
        """Parse context summary from LLM response"""
```

## 3. Implementierungsreihenfolge

### Phase 1: Core Infrastructure (2-3h)

#### 3.1 Data Models & Configuration
```python
# 1. core/chunking/chunk_models.py (45min)
- ContextualChunk, ContextGroup, ChunkingResult
- Enums: ChunkType, ContextGroupType, ChunkingStrategy
- Helper classes: ChunkPosition, StructureInfo, ChunkingStats

# 2. config/chunking_config.yaml (15min)
- Chunking strategies per document type
- Context inheritance settings
- Token limits and overlap ratios
- LLM settings for context generation

# 3. Integration in core/config.py (15min)
- ChunkingConfig dataclass
- Load chunking configuration
```

#### 3.2 Base Chunking Logic
```python
# 4. core/chunking/base_chunker.py (45min)
- Abstract BaseChunker class
- Token counting utilities
- Segment-to-Chunk conversion logic
- Visual element integration

# 5. core/chunking/strategies/structure_aware.py (60min)
- Structure-aware chunking using existing segments
- Respect document boundaries (headings, tables, visual elements)
- Multi-modal content preservation
```

### Phase 2: Context Group Formation (1-2h)

#### 3.3 Context Grouping
```python
# 6. core/chunking/context_grouper.py (90min)
- Document type-specific grouping strategies
- PDF: SmolDocling structure analysis
- DOCX: Heading hierarchy detection
- XLSX: Sheet and data block grouping
- PPTX: Topic-based slide grouping
- Semantic boundary detection (optional)
```

### Phase 3: Context Inheritance (2-3h)

#### 3.4 Context Summary Generation
```python
# 7. core/chunking/context_summarizer.py (120min)
- Integration with Hochschul-LLM client
- Dual-task prompt generation
- Context summary extraction and validation
- Context quality scoring
- Error handling and fallbacks

# 8. Prompt templates (30min)
- plugins/templates/context_generation.txt
- plugins/templates/task_with_context.txt
- Multi-language support (DE/EN)
```

#### 3.5 Context Propagation
```python
# 9. Context inheritance workflow (60min)
- Context propagation through chunk groups
- Context decay and refresh strategies
- Context relevance scoring
- Context chain management
```

### Phase 4: Main Chunker Integration (1h)

#### 3.6 Main Chunker Class
```python
# 10. core/content_chunker.py (60min)
- ContentChunker main class
- Integration of all components
- Chunking workflow orchestration
- Error handling and recovery
- Performance optimization
```

### Phase 5: Testing & Integration (1-2h)

#### 3.7 Tests & Examples
```python
# 11. tests/test_chunking.py (60min)
- Unit tests for all components
- Integration tests with real documents
- Performance benchmarks
- Context inheritance validation

# 12. examples/chunking_example.py (30min)
- End-to-end example
- Different document types
- Context inheritance demonstration
```

## 4. Technische Implementierungsdetails

### 4.1 Token Management
```python
# Token counting strategy
def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Accurate token estimation using tiktoken"""
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Context budget allocation
total_budget = 2000  # tokens per chunk
context_budget = 300  # max tokens for inherited context
content_budget = 1700  # remaining for actual content
```

### 4.2 Context Group Detection
```python
# PDF structure detection using SmolDocling data
def detect_pdf_sections(document: Document) -> List[Section]:
    """Use SmolDocling structured output for section detection"""
    sections = []
    current_section = None
    
    for segment in document.segments:
        # Check SmolDocling metadata for structure info
        if segment.metadata.get("smoldocling_element_type") == "text":
            if segment.segment_type == "heading":
                # Start new section
                current_section = Section(
                    title=segment.content,
                    start_segment=segment.segment_index,
                    level=detect_heading_level(segment)
                )
                sections.append(current_section)
            elif current_section:
                current_section.end_segment = segment.segment_index
    
    return sections
```

### 4.3 Multi-Modal Context Integration
```python
# Visual context in summaries
def integrate_visual_context(chunk: ContextualChunk) -> str:
    """Integrate visual element descriptions into context"""
    content_parts = [chunk.content]
    
    for visual in chunk.visual_elements:
        if visual.vlm_description:
            visual_context = f"[VISUAL: {visual.vlm_description}]"
            content_parts.append(visual_context)
            
        if visual.extracted_data:
            data_context = f"[DATA: {format_structured_data(visual.extracted_data)}]"
            content_parts.append(data_context)
    
    return "\n\n".join(content_parts)
```

### 4.4 Context Summary Prompt Template
```python
# Context generation prompt
CONTEXT_GENERATION_PROMPT = """
Du analysierst einen Textabschnitt und führst zwei Aufgaben aus:

HAUPTAUFGABE: {main_task}

ZUSÄTZLICHE AUFGABE: Erstelle eine Kontextzusammenfassung für nachfolgende Abschnitte.

Die Kontextzusammenfassung soll:
1. Kernthemen und wichtige Konzepte erfassen (2-3 Hauptpunkte)
2. Relevante Definitionen und Erklärungen enthalten
3. Wichtige Referenzen und Verweise festhalten
4. Den thematischen "roten Faden" bewahren
5. Maximal 300 Tokens umfassen

{previous_context_section}

AKTUELLER TEXTABSCHNITT:
{chunk_content}

{visual_elements_section}

AUSGABEFORMAT:
HAUPTAUFGABE ERGEBNIS:
[Hier das Ergebnis der Hauptaufgabe]

KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:
[Kompakte Zusammenfassung der relevanten Kontextinformationen]
"""
```

### 4.5 Performance Optimizations
```python
# Asynchrone Verarbeitung
async def process_context_groups_parallel(groups: List[ContextGroup]) -> List[ContextGroup]:
    """Process multiple context groups in parallel"""
    tasks = []
    for group in groups:
        task = asyncio.create_task(process_single_context_group(group))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]

# Context caching
class ContextCache:
    """Cache context summaries to avoid regeneration"""
    def __init__(self):
        self._cache = {}
    
    def get_context(self, chunk_id: str, task_type: str) -> Optional[str]:
        return self._cache.get(f"{chunk_id}:{task_type}")
    
    def set_context(self, chunk_id: str, task_type: str, context: str):
        self._cache[f"{chunk_id}:{task_type}"] = context
```

## 5. Konfiguration

### 5.1 Chunking Configuration
```yaml
# config/chunking.yaml
chunking:
  strategies:
    pdf:
      method: "structure_aware"
      max_tokens: 2000
      context_inheritance: true
      group_by: "sections"
      max_group_size: 8
      
    docx:
      method: "structure_aware"
      max_tokens: 1500
      context_inheritance: true
      group_by: "headings"
      respect_tables: true
      
    xlsx:
      method: "sheet_aware"
      max_tokens: 1000
      context_inheritance: true
      group_by: "sheets"
      
    pptx:
      method: "slide_aware"
      max_tokens: 800
      context_inheritance: true
      group_by: "topics"

  context_inheritance:
    enabled: true
    max_context_tokens: 300
    context_decay_rate: 0.1
    refresh_after_chunks: 6
    llm_model: "hochschul-llm"
    
  visual_integration:
    include_vlm_descriptions: true
    include_extracted_data: true
    max_visuals_per_chunk: 3
```

### 5.2 LLM Integration
```python
# Integration mit bestehendem Hochschul-LLM Client
class ContextSummarizer:
    def __init__(self, config: ChunkingConfig):
        self.llm_client = HochschulLLMClient()
        self.config = config
    
    async def generate_context_summary(self, chunk, task_context):
        # Use existing Hochschul-LLM infrastructure
        response = await self.llm_client.generate_completion(
            prompt=self._build_context_prompt(chunk, task_context),
            max_tokens=self.config.context_inheritance.max_context_tokens,
            temperature=0.1
        )
        return self._extract_context_summary(response)
```

## 6. Integration in bestehende Architektur

### 6.1 Pipeline Integration
```python
# Workflow: Document → Parser → Chunker → Storage
async def document_processing_pipeline(file_path: Path, task_template: str):
    # 1. Parse document (already implemented)
    document = await parser_factory.parse_document(file_path)
    
    # 2. Chunk with context inheritance (NEW)
    chunker = ContentChunker(config.chunking)
    chunking_result = await chunker.chunk_document_with_context_inheritance(
        document=document,
        task_template=task_template
    )
    
    # 3. Process chunks for downstream tasks
    for chunk in chunking_result.contextual_chunks:
        # Each chunk now has inherited context
        enhanced_prompt = chunk.get_enhanced_prompt(task_template)
        # → Continue to Triple Extraction, etc.
```

### 6.2 API Integration
```python
# FastAPI endpoint for chunking
@router.post("/documents/{document_id}/chunk")
async def chunk_document(
    document_id: str,
    chunking_strategy: Optional[str] = None,
    enable_context_inheritance: bool = True
):
    document = get_parsed_document(document_id)
    
    chunker = ContentChunker(
        strategy=chunking_strategy,
        enable_context_inheritance=enable_context_inheritance
    )
    
    result = await chunker.chunk_document_with_context_inheritance(document)
    
    return {
        "document_id": document_id,
        "chunk_count": len(result.contextual_chunks),
        "context_groups": len(result.context_groups),
        "chunks": [chunk.to_dict() for chunk in result.contextual_chunks]
    }
```

## 7. Test-Strategie

### 7.1 Unit Tests
```python
# tests/test_chunking.py
def test_context_group_formation():
    """Test PDF section detection"""
    
def test_context_summary_generation():
    """Test LLM-based context generation"""
    
def test_context_inheritance():
    """Test context propagation through chunk groups"""
    
def test_multi_modal_integration():
    """Test visual element integration in contexts"""
    
def test_token_management():
    """Test token counting and budget allocation"""
```

### 7.2 Integration Tests
```python
def test_end_to_end_chunking():
    """Test complete chunking workflow with real documents"""
    
def test_different_document_types():
    """Test chunking across PDF, DOCX, XLSX, PPTX"""
    
def test_performance_benchmarks():
    """Test chunking performance and scalability"""
```

## 8. Erfolgs-Metriken

### 8.1 Quality Metrics
- **Context Coherence**: Semantic similarity zwischen Chunks einer Gruppe
- **Context Relevance**: Relevanz der inherited contexts für nachfolgende Chunks
- **Information Preservation**: Anteil der wichtigen Informationen in Context Summaries

### 8.2 Performance Metrics
- **Chunking Speed**: Chunks pro Sekunde
- **Token Efficiency**: Verhältnis Content/Context Tokens
- **Memory Usage**: RAM-Verbrauch bei Batch-Processing

### 8.3 Integration Metrics
- **API Response Time**: < 2s für Standard-Dokumente
- **Batch Processing**: > 100 Dokumente/Stunde
- **Error Rate**: < 5% bei Context Generation

## 9. Risiken & Mitigation

### 9.1 LLM Dependency Risk
**Problem**: Hochschul-LLM Ausfälle beeinträchtigen Context Generation
**Mitigation**: Fallback auf simples Overlapping, lokales Backup-LLM

### 9.2 Context Quality Risk
**Problem**: Schlechte Context Summaries verschlechtern nachfolgende Chunks
**Mitigation**: Context Quality Scoring, automatische Fallbacks

### 9.3 Performance Risk
**Problem**: Context Generation verlangsamt Chunking erheblich
**Mitigation**: Asynchrone Verarbeitung, Context Caching, Batch-Optimierung

## 10. Nächste Schritte nach Implementierung

1. **Integration mit Triple Extraction**: Context-enhanced Prompts für bessere Tripel
2. **RAG Integration**: Chunks als Basis für Vector Store und Knowledge Graph
3. **Quality Feedback Loop**: Bewertung der Context Quality durch nachgelagerte Tasks
4. **Advanced Strategies**: Semantic Chunking, Cross-Document Context

---

**Geschätzte Gesamtzeit: 8-10 Stunden**
**Priorität: Hoch** (Grundlage für alle nachgelagerten LLM-Tasks)
**Abhängigkeiten**: Hochschul-LLM Client (✅ bereits implementiert)

Ist dieser Plan detailliert genug? Soll ich mit der Implementierung beginnen oder möchtest du noch Anpassungen/Ergänzungen?